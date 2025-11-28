{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "oai-nr-ue-multi.name" -}}
{{- $top := index . 0 -}}
{{- $ueId := index . 1 -}}
{{- default (printf "%s-%s" $top.Chart.Name $ueId) $top.Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "oai-nr-ue-multi.fullname" -}}
{{- $top := index . 0 -}}
{{- $ueId := index . 1 -}}
{{- if $top.Values.fullnameOverride -}}
{{- $top.Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default $top.Chart.Name $top.Values.nameOverride -}}
{{- if contains $name $top.Release.Name -}}
{{- $top.Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s-%s" $top.Release.Name $name $ueId | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "oai-nr-ue-multi.chart" -}}
{{- printf "%s-%s" $.Chart.Name $.Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "oai-nr-ue-multi.labels" -}}
{{- $top := index . 0 -}}
helm.sh/chart: {{ include "oai-nr-ue-multi.chart" $top }}
{{ include "oai-nr-ue-multi.selectorLabels" . }}
{{- if $top.Chart.AppVersion }}
app.kubernetes.io/version: {{ $top.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ $top.Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "oai-nr-ue-multi.selectorLabels" -}}
{{- $top := index . 0 -}}
{{- $ueId := index . 1 -}}
app.kubernetes.io/name: {{ include "oai-nr-ue-multi.name" . }}
app.kubernetes.io/instance: {{ $top.Release.Name }}
# ueId: {{ $ueId }}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "oai-nr-ue-multi.serviceAccountName" -}}
{{- if $.Values.serviceAccount.create -}}
    {{ default (include "oai-nr-ue-multi.fullname" .) $.Values.serviceAccount.name }}
{{- else -}}
    {{ default "default" $.Values.serviceAccount.name }}
{{- end -}}
{{- end -}}


