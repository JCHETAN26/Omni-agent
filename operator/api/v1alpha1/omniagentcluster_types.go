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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AutoscalingMode selects who scales the gateway Deployment.
// +kubebuilder:validation:Enum=off;hpa;operator
type AutoscalingMode string

const (
	// AutoscalingOff leaves the gateway's replica count alone.
	AutoscalingOff AutoscalingMode = "off"
	// AutoscalingHPA reconciles a standard autoscaling/v2 HPA on gateway CPU.
	AutoscalingHPA AutoscalingMode = "hpa"
	// AutoscalingOperator scales the gateway from the control plane's live load
	// (in-flight calls per gateway replica), polled every few seconds.
	AutoscalingOperator AutoscalingMode = "operator"
)

// ControlPlaneSpec describes the gRPC control-plane tier the operator runs.
type ControlPlaneSpec struct {
	// Image of omni-controlplane.
	Image string `json:"image"`
	// +kubebuilder:validation:Enum=Always;IfNotPresent;Never
	// +kubebuilder:default=IfNotPresent
	ImagePullPolicy corev1.PullPolicy `json:"imagePullPolicy,omitempty"`
	// Control-plane nodes. Clients fail over between them.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=3
	Replicas int32 `json:"replicas,omitempty"`
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	// Label selector (key: value) of pods allowed to call the control plane.
	// Defaults to {"omniagent.io/controlplane-client": "true"}.
	// +optional
	ClientSelector map[string]string `json:"clientSelector,omitempty"`
}

// GatewaySpec points at the existing gateway (FastAPI) Deployment.
type GatewaySpec struct {
	// Name of the gateway Deployment in the same namespace.
	DeploymentName string `json:"deploymentName"`
	// +kubebuilder:default=8000
	Port int32 `json:"port,omitempty"`
}

// AutoscalingSpec scales the gateway Deployment.
type AutoscalingSpec struct {
	// +kubebuilder:default=off
	Mode AutoscalingMode `json:"mode,omitempty"`
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	MinReplicas int32 `json:"minReplicas,omitempty"`
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=4
	MaxReplicas int32 `json:"maxReplicas,omitempty"`
	// hpa mode: target average CPU utilisation (% of requests).
	// +kubebuilder:default=60
	TargetCPUUtilization int32 `json:"targetCPUUtilization,omitempty"`
	// operator mode: target in-flight gateway calls per gateway replica.
	// +kubebuilder:default=2
	TargetInflightPerReplica int32 `json:"targetInflightPerReplica,omitempty"`
	// Both modes: wait this long below target before scaling down.
	// +kubebuilder:default=60
	ScaleDownStabilizationSeconds int32 `json:"scaleDownStabilizationSeconds,omitempty"`
	// operator mode: how often to poll the control plane's load.
	// +kubebuilder:default=2
	PollSeconds int32 `json:"pollSeconds,omitempty"`
}

// OmniAgentClusterSpec defines the desired state of OmniAgentCluster.
type OmniAgentClusterSpec struct {
	ControlPlane ControlPlaneSpec `json:"controlPlane"`
	Gateway      GatewaySpec      `json:"gateway"`
	// +optional
	Autoscaling AutoscalingSpec `json:"autoscaling,omitempty"`
}

// OmniAgentClusterStatus defines the observed state of OmniAgentCluster.
type OmniAgentClusterStatus struct {
	// Ready control-plane nodes.
	ControlPlaneReady int32 `json:"controlPlaneReady,omitempty"`
	// Gateway replicas as last set or observed.
	GatewayReplicas int32 `json:"gatewayReplicas,omitempty"`
	// operator mode: replicas the last load sample called for.
	DesiredGatewayReplicas int32 `json:"desiredGatewayReplicas,omitempty"`
	// operator mode: summed in-flight gateway calls across control-plane nodes.
	ObservedInflight int64 `json:"observedInflight,omitempty"`
	// operator mode: summed Search calls/s across control-plane nodes.
	ObservedRPS string `json:"observedRPS,omitempty"`
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty"`
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.controlPlaneReady`
// +kubebuilder:printcolumn:name="Mode",type=string,JSONPath=`.spec.autoscaling.mode`
// +kubebuilder:printcolumn:name="Gateways",type=integer,JSONPath=`.status.gatewayReplicas`
// +kubebuilder:printcolumn:name="Inflight",type=integer,JSONPath=`.status.observedInflight`
// +kubebuilder:resource:shortName=oac

// OmniAgentCluster runs OmniAgent's gRPC control plane in front of a gateway
// Deployment and optionally autoscales that gateway.
type OmniAgentCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   OmniAgentClusterSpec   `json:"spec,omitempty"`
	Status OmniAgentClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// OmniAgentClusterList contains a list of OmniAgentCluster.
type OmniAgentClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []OmniAgentCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&OmniAgentCluster{}, &OmniAgentClusterList{})
}
