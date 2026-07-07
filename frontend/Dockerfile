# syntax=docker/dockerfile:1
# Build context must be the frontend/ directory:
#   docker build ./frontend
#   docker build -f frontend/Dockerfile ./frontend   (from repo root)
#
# Override the backend URL at build time:
#   docker build --build-arg NEXT_PUBLIC_API_URL=http://api:8000 ./frontend

# ── Stage 1: Production-only dependencies ─────────────────────────────────────
FROM node:22-alpine AS prod-deps
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ── Stage 2: Build Next.js ─────────────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app

# Install ALL deps (devDeps needed: typescript, tailwindcss, postcss, eslint…)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source (respects .dockerignore — node_modules/.next excluded)
COPY . .

# NEXT_PUBLIC_* vars are inlined at build time by Next.js's compiler.
# To point at a different backend, pass --build-arg at docker build time.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ── Stage 3: Minimal production runner ────────────────────────────────────────
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

# Principle of least privilege
RUN addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 nextjs

# Production node_modules (no devDeps) from stage 1
COPY --from=prod-deps /app/node_modules ./node_modules

# Build artifacts from stage 2
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/public          ./public
COPY --from=builder /app/package.json    ./package.json

# Bundled npm CLI (node:22-alpine) ships picomatch 4.0.3 — not used at runtime.
# Run Next.js directly via node instead of `npm start`.
RUN rm -rf /usr/local/lib/node_modules/npm

USER nextjs

EXPOSE 3000

CMD ["node", "node_modules/next/dist/bin/next", "start"]
