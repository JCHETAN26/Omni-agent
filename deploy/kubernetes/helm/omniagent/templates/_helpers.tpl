{{- define "omniagent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "omniagent.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "omniagent.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "omniagent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: omniagent
{{- end -}}

{{/* Selector labels for one component: include "omniagent.selector" (list . "backend") */}}
{{- define "omniagent.selector" -}}
{{- $root := index . 0 -}}
app.kubernetes.io/name: {{ include "omniagent.name" $root }}
app.kubernetes.io/instance: {{ $root.Release.Name }}
app.kubernetes.io/component: {{ index . 1 }}
{{- end -}}

{{- define "omniagent.componentLabels" -}}
{{ include "omniagent.labels" (index . 0) }}
app.kubernetes.io/component: {{ index . 1 }}
{{- end -}}

{{- define "omniagent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "omniagent.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "omniagent.secretName" -}}
{{- default (printf "%s-secrets" (include "omniagent.fullname" .)) .Values.secrets.existingSecret -}}
{{- end -}}

{{/* Datastore endpoints: bundled StatefulSet, external URL, or empty (in-process fallback). */}}
{{- define "omniagent.qdrantUrl" -}}
{{- if .Values.qdrant.enabled -}}http://{{ include "omniagent.fullname" . }}-qdrant:6333{{- else -}}{{ .Values.qdrant.externalUrl }}{{- end -}}
{{- end -}}

{{- define "omniagent.elasticsearchUrl" -}}
{{- if .Values.elasticsearch.enabled -}}http://{{ include "omniagent.fullname" . }}-elasticsearch:9200{{- else -}}{{ .Values.elasticsearch.externalUrl }}{{- end -}}
{{- end -}}

{{- define "omniagent.redisUrl" -}}
{{- if .Values.redis.enabled -}}redis://{{ include "omniagent.fullname" . }}-redis:6379/0{{- else -}}{{ .Values.redis.externalUrl }}{{- end -}}
{{- end -}}

{{- define "omniagent.tritonUrl" -}}
{{- if .Values.triton.enabled -}}{{ include "omniagent.fullname" . }}-triton:8001{{- else -}}{{ .Values.triton.externalUrl }}{{- end -}}
{{- end -}}

{{/* Env shared by the API pods and the seed Job. */}}
{{- define "omniagent.backendEnv" -}}
envFrom:
  - configMapRef:
      name: {{ include "omniagent.fullname" . }}-config
  - secretRef:
      name: {{ include "omniagent.secretName" . }}
env:
  - name: POD_NAME
    valueFrom: { fieldRef: { fieldPath: metadata.name } }
  {{- with .Values.backend.extraEnv }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
{{- end -}}

{{/* Kafka bootstrap servers: bundled broker or kafka.externalBrokers. */}}
{{- define "omniagent.kafkaBrokers" -}}
{{- if .Values.kafka.enabled -}}{{ include "omniagent.fullname" . }}-kafka:9092{{- else -}}{{ .Values.kafka.externalBrokers }}{{- end -}}
{{- end -}}

{{/* Triton HTTP (KServe v2) endpoint, used by the Go indexer for embeddings. */}}
{{- define "omniagent.tritonHttpUrl" -}}
{{- if .Values.triton.enabled -}}http://{{ include "omniagent.fullname" . }}-triton:8000{{- else -}}{{ .Values.triton.externalHttpUrl }}{{- end -}}
{{- end -}}

{{- define "omniagent.ingestionPodSecurity" }}
      automountServiceAccountToken: false
      securityContext:
        # distroless "nonroot" (65532), static Go binaries.
        runAsNonRoot: true
        runAsUser: 65532
        runAsGroup: 65532
        seccompProfile: { type: RuntimeDefault }
{{- end }}
{{- define "omniagent.ingestionContainerSecurity" }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: [ALL] }
{{- end }}
