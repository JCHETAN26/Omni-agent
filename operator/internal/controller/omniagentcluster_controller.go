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
	"fmt"
	"strconv"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	platformv1alpha1 "github.com/JCHETAN26/Omni-agent/operator/api/v1alpha1"
)

// OmniAgentClusterReconciler reconciles an OmniAgentCluster object.
type OmniAgentClusterReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder events.EventRecorder
	Load     LoadSource
	Now      func() time.Time

	mu      sync.Mutex
	history map[types.NamespacedName][]recommendation
}

// +kubebuilder:rbac:groups=platform.omniagent.io,resources=omniagentclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=platform.omniagent.io,resources=omniagentclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=platform.omniagent.io,resources=omniagentclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=policy,resources=poddisruptionbudgets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=networkpolicies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete

func (r *OmniAgentClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)
	var c platformv1alpha1.OmniAgentCluster
	if err := r.Get(ctx, req.NamespacedName, &c); err != nil {
		if apierrors.IsNotFound(err) {
			r.forget(req.NamespacedName)
			return ctrl.Result{}, nil // owned objects are garbage-collected
		}
		return ctrl.Result{}, err
	}

	var gateway appsv1.Deployment
	gwKey := types.NamespacedName{Namespace: c.Namespace, Name: c.Spec.Gateway.DeploymentName}
	if err := r.Get(ctx, gwKey, &gateway); err != nil {
		r.setCondition(&c, "Ready", metav1.ConditionFalse, "GatewayNotFound", err.Error())
		_ = r.Status().Update(ctx, &c)
		return ctrl.Result{RequeueAfter: 10 * time.Second}, client.IgnoreNotFound(err)
	}

	// Desired child objects: create or bring back to spec (self-healing).
	for _, obj := range []struct {
		desired client.Object
		mutate  func(existing client.Object)
	}{
		{desiredControlPlane(&c), func(e client.Object) {
			d, want := e.(*appsv1.Deployment), desiredControlPlane(&c)
			d.Labels, d.Spec = want.Labels, want.Spec
		}},
		{desiredControlPlaneService(&c), func(e client.Object) {
			s, want := e.(*corev1.Service), desiredControlPlaneService(&c)
			s.Labels, s.Spec.Selector, s.Spec.Ports = want.Labels, want.Spec.Selector, want.Spec.Ports
			s.Spec.ClusterIP = corev1.ClusterIPNone
		}},
		{desiredGatewayEndpoints(&c, gateway.Spec.Selector.MatchLabels), func(e client.Object) {
			s, want := e.(*corev1.Service), desiredGatewayEndpoints(&c, gateway.Spec.Selector.MatchLabels)
			s.Labels, s.Spec.Selector, s.Spec.Ports = want.Labels, want.Spec.Selector, want.Spec.Ports
			s.Spec.ClusterIP, s.Spec.PublishNotReadyAddresses = corev1.ClusterIPNone, true
		}},
		{desiredPDB(&c), func(e client.Object) {
			p, want := e.(*policyv1.PodDisruptionBudget), desiredPDB(&c)
			p.Labels, p.Spec = want.Labels, want.Spec
		}},
		{desiredNetworkPolicy(&c), func(e client.Object) {
			n, want := e.(*networkingv1.NetworkPolicy), desiredNetworkPolicy(&c)
			n.Labels, n.Spec = want.Labels, want.Spec
		}},
	} {
		existing := obj.desired.DeepCopyObject().(client.Object)
		op, err := controllerutil.CreateOrUpdate(ctx, r.Client, existing, func() error {
			obj.mutate(existing)
			return controllerutil.SetControllerReference(&c, existing, r.Scheme)
		})
		if err != nil {
			return ctrl.Result{}, fmt.Errorf("%T %s: %w", existing, existing.GetName(), err)
		}
		if op != controllerutil.OperationResultNone {
			log.Info("reconciled", "kind", fmt.Sprintf("%T", existing), "name", existing.GetName(), "op", op)
		}
	}

	result, err := r.reconcileAutoscaling(ctx, &c, &gateway)
	if err != nil {
		return result, err
	}

	var cp appsv1.Deployment
	if err := r.Get(ctx, types.NamespacedName{Namespace: c.Namespace, Name: controlPlaneName(&c)}, &cp); err == nil {
		c.Status.ControlPlaneReady = cp.Status.ReadyReplicas
	}
	if c.Status.ControlPlaneReady >= max(c.Spec.ControlPlane.Replicas, 1) {
		r.setCondition(&c, "Ready", metav1.ConditionTrue, "ControlPlaneReady",
			fmt.Sprintf("%d control-plane nodes ready", c.Status.ControlPlaneReady))
	} else {
		r.setCondition(&c, "Ready", metav1.ConditionFalse, "ControlPlaneStarting",
			fmt.Sprintf("%d/%d control-plane nodes ready", c.Status.ControlPlaneReady, max(c.Spec.ControlPlane.Replicas, 1)))
	}
	if err := r.Status().Update(ctx, &c); err != nil && !apierrors.IsConflict(err) {
		return ctrl.Result{}, err
	}
	return result, nil
}

func (r *OmniAgentClusterReconciler) reconcileAutoscaling(ctx context.Context, c *platformv1alpha1.OmniAgentCluster,
	gateway *appsv1.Deployment) (ctrl.Result, error) {
	a := c.Spec.Autoscaling
	hpa := &autoscalingv2.HorizontalPodAutoscaler{}
	hpaKey := types.NamespacedName{Namespace: c.Namespace, Name: hpaName(c)}
	if a.Mode == platformv1alpha1.AutoscalingHPA {
		existing := desiredHPA(c)
		_, err := controllerutil.CreateOrUpdate(ctx, r.Client, existing, func() error {
			want := desiredHPA(c)
			existing.Labels, existing.Spec = want.Labels, want.Spec
			return controllerutil.SetControllerReference(c, existing, r.Scheme)
		})
		c.Status.GatewayReplicas = gateway.Status.Replicas
		return ctrl.Result{}, err
	}
	// Not hpa mode: remove an HPA we created earlier so it cannot fight us.
	if err := r.Get(ctx, hpaKey, hpa); err == nil && metav1.IsControlledBy(hpa, c) {
		if err := r.Delete(ctx, hpa); client.IgnoreNotFound(err) != nil {
			return ctrl.Result{}, err
		}
	}
	c.Status.GatewayReplicas = ptrValue(gateway.Spec.Replicas)
	if a.Mode != platformv1alpha1.AutoscalingOperator {
		return ctrl.Result{}, nil
	}

	poll := time.Duration(max(a.PollSeconds, 1)) * time.Second
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(c.Namespace), client.MatchingLabels(controlPlaneLabels(c))); err != nil {
		return ctrl.Result{}, err
	}
	inflight, rps, err := r.Load.Load(ctx, pods.Items)
	if err != nil {
		// No load signal: hold the current size rather than guess.
		logf.FromContext(ctx).Info("control-plane load unavailable; holding gateway size", "err", err.Error())
		return ctrl.Result{RequeueAfter: poll}, nil
	}
	key := types.NamespacedName{Namespace: c.Namespace, Name: c.Name}
	now := r.now()
	current := ptrValue(gateway.Spec.Replicas)
	r.mu.Lock()
	if r.history == nil {
		r.history = map[types.NamespacedName][]recommendation{}
	}
	desired, hist := decide(now, current, a.MinReplicas, a.MaxReplicas, inflight, a.TargetInflightPerReplica,
		r.history[key], time.Duration(a.ScaleDownStabilizationSeconds)*time.Second)
	r.history[key] = hist
	r.mu.Unlock()
	c.Status.ObservedInflight = int64(inflight + 0.5)
	c.Status.ObservedRPS = strconv.FormatFloat(rps, 'f', 1, 64)
	c.Status.DesiredGatewayReplicas = desired
	if desired != current {
		patch := client.MergeFrom(gateway.DeepCopy())
		gateway.Spec.Replicas = &desired
		if err := r.Patch(ctx, gateway, patch); err != nil {
			return ctrl.Result{}, err
		}
		c.Status.LastScaleTime = &metav1.Time{Time: now}
		c.Status.GatewayReplicas = desired
		logf.FromContext(ctx).Info("scaled gateway", "from", current, "to", desired, "inflight", inflight, "rps", rps)
		if r.Recorder != nil {
			r.Recorder.Eventf(c, gateway, corev1.EventTypeNormal, "ScaledGateway", "Scale",
				"gateway %d -> %d replicas (in-flight %.1f, %.1f rps)", current, desired, inflight, rps)
		}
	}
	return ctrl.Result{RequeueAfter: poll}, nil
}

func (r *OmniAgentClusterReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

func (r *OmniAgentClusterReconciler) forget(key types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.history, key)
}

func (r *OmniAgentClusterReconciler) setCondition(c *platformv1alpha1.OmniAgentCluster, t string,
	s metav1.ConditionStatus, reason, msg string) {
	meta.SetStatusCondition(&c.Status.Conditions, metav1.Condition{
		Type: t, Status: s, Reason: reason, Message: msg, ObservedGeneration: c.Generation,
	})
}

func ptrValue(p *int32) int32 {
	if p == nil {
		return 1
	}
	return *p
}

// SetupWithManager sets up the controller with the Manager.
func (r *OmniAgentClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&platformv1alpha1.OmniAgentCluster{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Owns(&policyv1.PodDisruptionBudget{}).
		Owns(&networkingv1.NetworkPolicy{}).
		Owns(&autoscalingv2.HorizontalPodAutoscaler{}).
		Named("omniagentcluster").
		Complete(r)
}
