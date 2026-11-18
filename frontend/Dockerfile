# syntax=docker/dockerfile:1.7
# OmniAgent UI: static Vite build served by unprivileged nginx (port 8080).
#   docker build -t omniagent-frontend:dev frontend/
FROM node:22-alpine AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30.6-alpine
# Templates in /etc/nginx/templates are rendered with envsubst at startup.
ENV BACKEND_URL=http://omniagent-backend:8000
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 8080
