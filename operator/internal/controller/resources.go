package controller

import (
	"fmt"
	"maps"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	platformv1alpha1 "github.com/JCHETAN26/Omni-agent/operator/api/v1alpha1"
)

const (
	grpcPort    = 9000
	metricsPort = 9090
	// OperatorPodLabel selects the operator's own pod (the scaffold's manager
	// label): it polls the control plane's Stats RPC for autoscaling.
	OperatorPodLabel = "control-plane"
	operatorPodValue = "controller-manager"
)

// Names of the objects an OmniAgentCluster owns.
func controlPlaneName(c *platformv1alpha1.OmniAgentCluster) string { return c.Name + "-controlplane" }
func gatewayEndpointsName(c *platformv1alpha1.OmniAgentCluster) string {
	return c.Name + "-gateway-endpoints"
}
func hpaName(c *platformv1alpha1.OmniAgentCluster) string { return c.Name + "-gateway" }

func controlPlaneLabels(c *platformv1alpha1.OmniAgentCluster) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "omniagent",
		"app.kubernetes.io/component":  "controlplane",
		"app.kubernetes.io/instance":   c.Name,
		"app.kubernetes.io/managed-by": "omniagent-operator",
	}
}

func clientSelector(c *platformv1alpha1.OmniAgentCluster) map[string]string {
	if len(c.Spec.ControlPlane.ClientSelector) > 0 {
		return c.Spec.ControlPlane.ClientSelector
	}
	return map[string]string{"omniagent.io/controlplane-client": "true"}
}

func defaultResources(r corev1.ResourceRequirements) corev1.ResourceRequirements {
	if r.Requests == nil && r.Limits == nil {
		return corev1.ResourceRequirements{
			Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("50m"), corev1.ResourceMemory: resource.MustParse("32Mi")},
			Limits:   corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("500m"), corev1.ResourceMemory: resource.MustParse("128Mi")},
		}
	}
	return r
}

// desiredControlPlane is the control-plane Deployment: distroless non-root,
// read-only root filesystem, no capabilities, gRPC health probes.
func desiredControlPlane(c *platformv1alpha1.OmniAgentCluster) *appsv1.Deployment {
	labels := controlPlaneLabels(c)
	cp := c.Spec.ControlPlane
	probe := func(period int32) *corev1.Probe {
		return &corev1.Probe{
			ProbeHandler:  corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: grpcPort, Service: ptr.To("omniagent.controlplane.v1.ControlPlane")}},
			PeriodSeconds: period,
		}
	}
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: controlPlaneName(c), Namespace: c.Namespace, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(max(cp.Replicas, 1)),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxUnavailable: ptr.To(intstr.FromInt32(0)), MaxSurge: ptr.To(intstr.FromInt32(1)),
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels, Annotations: map[string]string{
					"prometheus.io/scrape": "true", "prometheus.io/port": fmt.Sprint(metricsPort),
				}},
				Spec: corev1.PodSpec{
					AutomountServiceAccountToken: ptr.To(false),
					// SIGTERM: report NOT_SERVING for DRAIN_SECONDS, then drain in-flight calls.
					TerminationGracePeriodSeconds: ptr.To[int64](30),
					SecurityContext: &corev1.PodSecurityContext{
						RunAsNonRoot:   ptr.To(true),
						RunAsUser:      ptr.To[int64](65532),
						RunAsGroup:     ptr.To[int64](65532),
						SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault},
					},
					// Spread nodes so one Kubernetes node failure leaves the others.
					TopologySpreadConstraints: []corev1.TopologySpreadConstraint{{
						MaxSkew: 1, TopologyKey: "kubernetes.io/hostname",
						WhenUnsatisfiable: corev1.ScheduleAnyway,
						LabelSelector:     &metav1.LabelSelector{MatchLabels: labels},
					}},
					Containers: []corev1.Container{{
						Name:            "controlplane",
						Image:           cp.Image,
						ImagePullPolicy: cp.ImagePullPolicy,
						Command:         []string{"/omni-controlplane"},
						Env: []corev1.EnvVar{
							{Name: "GATEWAY_HOST", Value: gatewayEndpointsName(c)},
							{Name: "GATEWAY_PORT", Value: fmt.Sprint(gatewayPort(c))},
							{Name: "NODE_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
						},
						Ports: []corev1.ContainerPort{
							{Name: "grpc", ContainerPort: grpcPort},
							{Name: "metrics", ContainerPort: metricsPort},
						},
						ReadinessProbe: probe(2),
						LivenessProbe:  probe(10),
						Resources:      defaultResources(cp.Resources),
						SecurityContext: &corev1.SecurityContext{
							AllowPrivilegeEscalation: ptr.To(false),
							ReadOnlyRootFilesystem:   ptr.To(true),
							Capabilities:             &corev1.Capabilities{Drop: []corev1.Capability{"ALL"}},
						},
					}},
				},
			},
		},
	}
}

func gatewayPort(c *platformv1alpha1.OmniAgentCluster) int32 {
	if c.Spec.Gateway.Port == 0 {
		return 8000
	}
	return c.Spec.Gateway.Port
}

// desiredControlPlaneService is headless: clients resolve every node and
// balance/fail over across them (round_robin + health checks).
func desiredControlPlaneService(c *platformv1alpha1.OmniAgentCluster) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: controlPlaneName(c), Namespace: c.Namespace, Labels: controlPlaneLabels(c)},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  controlPlaneLabels(c),
			Ports: []corev1.ServicePort{
				{Name: "grpc", Port: grpcPort, TargetPort: intstr.FromString("grpc")},
				{Name: "metrics", Port: metricsPort, TargetPort: intstr.FromString("metrics")},
			},
		},
	}
}

// desiredGatewayEndpoints is a headless Service over the gateway pods, so each
// control-plane node sees (and health-checks) every gateway individually.
func desiredGatewayEndpoints(c *platformv1alpha1.OmniAgentCluster, gatewaySelector map[string]string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: gatewayEndpointsName(c), Namespace: c.Namespace, Labels: controlPlaneLabels(c)},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  maps.Clone(gatewaySelector),
			// Probing decides health: list gateways even before they are Ready.
			PublishNotReadyAddresses: true,
			Ports:                    []corev1.ServicePort{{Name: "http", Port: gatewayPort(c), TargetPort: intstr.FromInt32(gatewayPort(c))}},
		},
	}
}

func desiredPDB(c *platformv1alpha1.OmniAgentCluster) *policyv1.PodDisruptionBudget {
	return &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: controlPlaneName(c), Namespace: c.Namespace, Labels: controlPlaneLabels(c)},
		Spec: policyv1.PodDisruptionBudgetSpec{
			MaxUnavailable: ptr.To(intstr.FromInt32(1)),
			Selector:       &metav1.LabelSelector{MatchLabels: controlPlaneLabels(c)},
		},
	}
}

// desiredNetworkPolicy admits gRPC only from labelled clients and the
// operator, and metrics only from Prometheus in the monitoring namespace.
func desiredNetworkPolicy(c *platformv1alpha1.OmniAgentCluster) *networkingv1.NetworkPolicy {
	tcp := corev1.ProtocolTCP
	port := func(p int32) networkingv1.NetworkPolicyPort {
		return networkingv1.NetworkPolicyPort{Protocol: &tcp, Port: ptr.To(intstr.FromInt32(p))}
	}
	return &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: controlPlaneName(c), Namespace: c.Namespace, Labels: controlPlaneLabels(c)},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{MatchLabels: controlPlaneLabels(c)},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					From: []networkingv1.NetworkPolicyPeer{
						{PodSelector: &metav1.LabelSelector{MatchLabels: clientSelector(c)}},
						{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{OperatorPodLabel: operatorPodValue}},
							NamespaceSelector: &metav1.LabelSelector{}},
					},
					Ports: []networkingv1.NetworkPolicyPort{port(grpcPort)},
				},
				{
					From: []networkingv1.NetworkPolicyPeer{{
						NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"kubernetes.io/metadata.name": "monitoring"}},
						PodSelector:       &metav1.LabelSelector{MatchLabels: map[string]string{"app.kubernetes.io/name": "prometheus"}},
					}},
					Ports: []networkingv1.NetworkPolicyPort{port(metricsPort)},
				},
			},
		},
	}
}

// desiredHPA is the "hpa" mode: a standard CPU HPA on the gateway, with the
// same bounds and scale-down stabilization as the operator mode.
func desiredHPA(c *platformv1alpha1.OmniAgentCluster) *autoscalingv2.HorizontalPodAutoscaler {
	a := c.Spec.Autoscaling
	return &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Name: hpaName(c), Namespace: c.Namespace, Labels: controlPlaneLabels(c)},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{APIVersion: "apps/v1", Kind: "Deployment", Name: c.Spec.Gateway.DeploymentName},
			MinReplicas:    ptr.To(max(a.MinReplicas, 1)),
			MaxReplicas:    max(a.MaxReplicas, max(a.MinReplicas, 1)),
			Metrics: []autoscalingv2.MetricSpec{{
				Type: autoscalingv2.ResourceMetricSourceType,
				Resource: &autoscalingv2.ResourceMetricSource{
					Name:   corev1.ResourceCPU,
					Target: autoscalingv2.MetricTarget{Type: autoscalingv2.UtilizationMetricType, AverageUtilization: ptr.To(a.TargetCPUUtilization)},
				},
			}},
			Behavior: &autoscalingv2.HorizontalPodAutoscalerBehavior{
				ScaleDown: &autoscalingv2.HPAScalingRules{StabilizationWindowSeconds: ptr.To(a.ScaleDownStabilizationSeconds)},
			},
		},
	}
}
