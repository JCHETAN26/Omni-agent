# Network Troubleshooting Guide

## Packet Loss on Edge Routers
Intermittent packet loss on edge routers is most often caused by MTU mismatches between the overlay network and the physical uplink. The overlay adds a 50-byte VXLAN header, so the physical MTU must be at least 1550 when the overlay MTU is 1500.

## TLS Handshake Failures
A `handshake failure` alert usually means the client and server share no cipher suite, or the server certificate chain is incomplete. Verify the chain with `openssl s_client -connect host:443 -showcerts` and confirm the intermediate certificate is served.

## DNS Resolution Slowness
Slow DNS lookups inside pods are typically caused by the `ndots:5` default, which makes the resolver try every search domain before the absolute name. Set `ndots:2` in the pod DNS config for external-heavy workloads.
