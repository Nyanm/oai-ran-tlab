{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "oai-gnb-multi.name" -}}
{{- $top := index . 0 -}}
{{- $gnbId := index . 1 -}}
{{- default (printf "%s-%s" $top.Chart.Name $gnbId) $top.Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "oai-gnb-multi.fullname" -}}
{{- $top := index . 0 -}}
{{- $gnbId := index . 1 -}}
{{- if $top.Values.fullnameOverride -}}
{{- $top.Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default $top.Chart.Name $top.Values.nameOverride -}}
{{- if contains $name $top.Release.Name -}}
{{- $top.Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s-%s" $top.Release.Name $name $gnbId | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "oai-gnb-multi.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "oai-gnb-multi.labels" -}}
{{- $top := index . 0 -}}
helm.sh/chart: {{ include "oai-gnb-multi.chart" $top }}
{{ include "oai-gnb-multi.selectorLabels" . }}
{{- if $top.Chart.AppVersion }}
app.kubernetes.io/version: {{ $top.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ $top.Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "oai-gnb-multi.selectorLabels" -}}
{{- $top := index . 0 -}}
app.kubernetes.io/name: {{ include "oai-gnb-multi.name" . }}
app.kubernetes.io/instance: {{ $top.Release.Name }}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "oai-gnb-multi.serviceAccountName" -}}
{{- $top := index . 0 -}}
{{- if $top.Values.serviceAccount.create -}}
    {{ default (include "oai-gnb-multi.fullname" .) $top.Values.serviceAccount.name }}
{{- else -}}
    {{ default "default" $top.Values.serviceAccount.name }}
{{- end -}}
{{- end -}}
