# Helm chart — video-understanding (minimal, portfolio scope)

Deploys only `api`, `worker`, `frontend` to K8s. Redis/Qdrant/Postgres/MinIO
keep running via the existing root `docker-compose.yml` on the host — adding a
Bitnami dependency chart for them was more effort (dependency management,
extra PVCs, values overrides) for no benefit at this scope, so the chart
instead points at them via `host.docker.internal` (same pattern already used
for `prometheus` in `docker-compose.yml`).

Not included on purpose (see task scope): Ingress/TLS, HPA, ServiceMonitor,
external-secrets, mlflow/prometheus/grafana/loki/tempo.

## Prerequisites

1. `docker-compose up -d redis qdrant postgres minio` running on the host.
2. A local cluster: `kind create cluster` or `minikube start`.
3. On kind/Docker Desktop, `host.docker.internal` resolves to the host from
   inside pods. On minikube (Linux driver), it may not — use
   `minikube start --extra-config` networking or override
   `config.redisUrl` / `config.qdrantHost` / `config.minioEndpoint` /
   `secrets.postgresUrl` with `minikube ssh` gateway IP (`192.168.49.1`)
   instead of `host.docker.internal`.
4. Ollama (`ollama serve`) running on the host — `config.ollamaHost` needs the
   same host resolution as above.

## Try it locally

```bash
helm lint deploy/helm/video-understanding
helm template deploy/helm/video-understanding

# install (uses default "scan" tag, pushed on every build.yml run)
helm install vu deploy/helm/video-understanding \
  --set image.tag=sha-XXXXXXX   # or a real vX.Y.Z tag from a release

# port-forward to test
kubectl port-forward svc/vu-frontend 3000:3000
kubectl port-forward svc/vu-api 8000:8000

helm uninstall vu
```

To enable the GPU node selector/resources for the worker: `--set gpu.enabled=true`
(requires a real node with `nvidia.com/gpu` capacity, e.g. NVIDIA device plugin —
not applicable to plain kind/minikube without extra setup).

## Known limitation

`NEXT_PUBLIC_API_URL` is inlined into the frontend image at build time by
Next.js (see `frontend/Dockerfile`). Setting it via this chart has no runtime
effect — it only reflects what CI baked in when the image was built.
