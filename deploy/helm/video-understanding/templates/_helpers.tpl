{{- define "video-understanding.fullname" -}}
{{ .Release.Name }}
{{- end -}}

{{- define "video-understanding.image" -}}
{{ .Values.image.registry }}/{{ .Values.image.repository }}/{{ .name }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "video-understanding.labels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: video-understanding
{{- end -}}
