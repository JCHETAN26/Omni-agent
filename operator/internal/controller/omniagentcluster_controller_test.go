/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	platformv1alpha1 "github.com/JCHETAN26/Omni-agent/operator/api/v1alpha1"
)

// fakeLoad returns a fixed control-plane load.
type fakeLoad struct{ inflight float64 }

func (f *fakeLoad) Load(context.Context, []corev1.Pod) (float64, float64, error) {
	return f.inflight, f.inflight * 5, nil
}

var _ = Describe("OmniAgentCluster controller", func() {
	const ns = "default"
	var (
		load       *fakeLoad
		clock      time.Time
		reconciler *OmniAgentClusterReconciler
		key        = types.NamespacedName{Namespace: ns, Name: "omni"}
	)
	gatewayLabels := map[string]string{"app.kubernetes.io/component": "backend"}

	reconcileOnce := func() reconcile.Result {
		res, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		Expect(err).NotTo(HaveOccurred())
		return res
	}
	gatewayReplicas := func() int32 {
		var d appsv1.Deployment
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "gateway"}, &d)).To(Succeed())
		return *d.Spec.Replicas
	}
	setMode := func(mode platformv1alpha1.AutoscalingMode) {
		var c platformv1alpha1.OmniAgentCluster
		Expect(k8sClient.Get(ctx, key, &c)).To(Succeed())
		c.Spec.Autoscaling.Mode = mode
		Expect(k8sClient.Update(ctx, &c)).To(Succeed())
	}

	BeforeEach(func() {
		load = &fakeLoad{}
		clock = time.Unix(1_800_000_000, 0)
		reconciler = &OmniAgentClusterReconciler{
			Client: k8sClient, Scheme: scheme.Scheme, Load: load, Now: func() time.Time { return clock },
		}
		gateway := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "gateway", Namespace: ns},
			Spec: appsv1.DeploymentSpec{
				Replicas: ptr.To[int32](1),
				Selector: &metav1.LabelSelector{MatchLabels: gatewayLabels},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: gatewayLabels},
					Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "api", Image: "gateway:1"}}},
				},
			},
		}
		Expect(k8sClient.Create(ctx, gateway)).To(Succeed())
		cr := &platformv1alpha1.OmniAgentCluster{
			ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: ns},
			Spec: platformv1alpha1.OmniAgentClusterSpec{
				ControlPlane: platformv1alpha1.ControlPlaneSpec{Image: "omniagent-controlplane:dev", Replicas: 3},
				Gateway:      platformv1alpha1.GatewaySpec{DeploymentName: "gateway"},
				Autoscaling: platformv1alpha1.AutoscalingSpec{
					Mode: platformv1alpha1.AutoscalingOff, MinReplicas: 1, MaxReplicas: 4,
					TargetInflightPerReplica: 2, ScaleDownStabilizationSeconds: 60, TargetCPUUtilization: 60, PollSeconds: 2,
				},
			},
		}
		Expect(k8sClient.Create(ctx, cr)).To(Succeed())
	})

	AfterEach(func() {
		_ = k8sClient.Delete(ctx, &platformv1alpha1.OmniAgentCluster{ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: ns}})
		_ = k8sClient.Delete(ctx, &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "gateway", Namespace: ns}})
		// envtest has no garbage collector: remove owned objects by hand.
		_ = k8sClient.Delete(ctx, &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "omni-controlplane", Namespace: ns}})
		for _, n := range []string{"omni-controlplane", "omni-gateway-endpoints"} {
			_ = k8sClient.Delete(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: n, Namespace: ns}})
		}
		_ = k8sClient.Delete(ctx, &policyv1.PodDisruptionBudget{ObjectMeta: metav1.ObjectMeta{Name: "omni-controlplane", Namespace: ns}})
		_ = k8sClient.Delete(ctx, &networkingv1.NetworkPolicy{ObjectMeta: metav1.ObjectMeta{Name: "omni-controlplane", Namespace: ns}})
		_ = k8sClient.Delete(ctx, &autoscalingv2.HorizontalPodAutoscaler{ObjectMeta: metav1.ObjectMeta{Name: "omni-gateway", Namespace: ns}})
	})

	It("provisions a hardened control plane owned by the resource", func() {
		reconcileOnce()
		var d appsv1.Deployment
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "omni-controlplane"}, &d)).To(Succeed())
		Expect(*d.Spec.Replicas).To(Equal(int32(3)))
		Expect(d.OwnerReferences).To(HaveLen(1))
		Expect(d.OwnerReferences[0].Kind).To(Equal("OmniAgentCluster"))
		pod := d.Spec.Template.Spec
		Expect(*pod.SecurityContext.RunAsNonRoot).To(BeTrue())
		c := pod.Containers[0]
		Expect(*c.SecurityContext.ReadOnlyRootFilesystem).To(BeTrue())
		Expect(c.SecurityContext.Capabilities.Drop).To(ConsistOf(corev1.Capability("ALL")))
		Expect(c.ReadinessProbe.GRPC).NotTo(BeNil())

		var gw corev1.Service
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "omni-gateway-endpoints"}, &gw)).To(Succeed())
		Expect(gw.Spec.ClusterIP).To(Equal(corev1.ClusterIPNone))
		Expect(gw.Spec.Selector).To(Equal(gatewayLabels)) // copied from the gateway Deployment
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "omni-controlplane"}, &policyv1.PodDisruptionBudget{})).To(Succeed())
		var np networkingv1.NetworkPolicy
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "omni-controlplane"}, &np)).To(Succeed())
		Expect(np.Spec.Ingress).To(HaveLen(2))
	})

	It("restores drifted or deleted children (self-healing)", func() {
		reconcileOnce()
		dkey := types.NamespacedName{Namespace: ns, Name: "omni-controlplane"}
		var d appsv1.Deployment
		Expect(k8sClient.Get(ctx, dkey, &d)).To(Succeed())
		d.Spec.Template.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = ptr.To(false)
		d.Spec.Replicas = ptr.To[int32](1)
		Expect(k8sClient.Update(ctx, &d)).To(Succeed())
		reconcileOnce()
		Expect(k8sClient.Get(ctx, dkey, &d)).To(Succeed())
		Expect(*d.Spec.Replicas).To(Equal(int32(3)))
		Expect(*d.Spec.Template.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem).To(BeTrue())

		Expect(k8sClient.Delete(ctx, &d)).To(Succeed())
		Eventually(func() bool { return apierrors.IsNotFound(k8sClient.Get(ctx, dkey, &appsv1.Deployment{})) }).Should(BeTrue())
		reconcileOnce()
		Expect(k8sClient.Get(ctx, dkey, &d)).To(Succeed())
	})

	It("owns an HPA only in hpa mode", func() {
		setMode(platformv1alpha1.AutoscalingHPA)
		reconcileOnce()
		hkey := types.NamespacedName{Namespace: ns, Name: "omni-gateway"}
		var h autoscalingv2.HorizontalPodAutoscaler
		Expect(k8sClient.Get(ctx, hkey, &h)).To(Succeed())
		Expect(h.Spec.ScaleTargetRef.Name).To(Equal("gateway"))
		Expect(*h.Spec.Behavior.ScaleDown.StabilizationWindowSeconds).To(Equal(int32(60)))

		setMode(platformv1alpha1.AutoscalingOperator)
		reconcileOnce()
		Expect(apierrors.IsNotFound(k8sClient.Get(ctx, hkey, &h))).To(BeTrue())
	})

	It("scales the gateway from control-plane load, holding capacity through the stabilization window", func() {
		setMode(platformv1alpha1.AutoscalingOperator)
		load.inflight = 7 // ceil(7/2) = 4 replicas
		res := reconcileOnce()
		Expect(res.RequeueAfter).To(Equal(2 * time.Second))
		Expect(gatewayReplicas()).To(Equal(int32(4)))

		load.inflight = 20 // capped at maxReplicas
		reconcileOnce()
		Expect(gatewayReplicas()).To(Equal(int32(4)))

		load.inflight = 0
		clock = clock.Add(30 * time.Second)
		reconcileOnce()
		Expect(gatewayReplicas()).To(Equal(int32(4)), "scaled down inside the stabilization window")

		clock = clock.Add(31 * time.Second)
		reconcileOnce()
		Expect(gatewayReplicas()).To(Equal(int32(1)))

		var c platformv1alpha1.OmniAgentCluster
		Expect(k8sClient.Get(ctx, key, &c)).To(Succeed())
		Expect(c.Status.DesiredGatewayReplicas).To(Equal(int32(1)))
		Expect(c.Status.LastScaleTime).NotTo(BeNil())
	})
})
