# Phase 1: Foundation Implementation

## Goal
Implement the core infrastructure and skeleton services for the 7-Layer Sentinel.

## Tasks
- [ ] **Infrastructure**: Create `envoy.yaml` and `docker-compose.yaml` to wire up the sidecar pattern.
- [ ] **Layer 1**: Implement `RuntimeIdentityService.java` (Mock) to pass simple identity checks.
- [ ] **Layer 3**: Implement `DeterministicRuleEngine.java` to block `/admin` paths.
- [ ] **Layer 5**: Implement `TrafficLightService.java` to orchestrate the flow (Layer 1 -> 3 -> 5).
- [ ] **Integration**: Update `VeilSecurityFilter.java` to use `TrafficLightService`.

## Verification
- Start the stack with `docker-compose up`.
- Curl port 10000 and verify traffic flows through VEIL.
- Verify `TrafficLightService` returns GREEN/RED based on mock rules.
