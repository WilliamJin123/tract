# Microservices vs Monolith: Data Consistency Trade-offs

**Focus Angle**: Data consistency guarantees, failure semantics, and schema evolution

## Monolith Advantages

### Strong Consistency (ACID Transactions)
- **Single database**: Multi-step operations execute atomically across tables
- Rollback on failure is guaranteed — partial writes impossible
- No distributed transaction coordination needed
- Example: Invoice system can debit account + create receipt in one transaction, both succeed or both fail

### Synchronous Feedback
- Operation results known immediately
- Client validation happens at DB constraint layer
- Schema changes impact one database, one migration timeline

### Simplified Reasoning
- One consistency model across the entire application
- Cascading deletes, foreign keys, triggers work globally
- Developers familiar with relational ACID semantics

## Microservices Challenges

### Distributed Consistency Problem
- Each service owns its database → no shared transactions
- Order service commits, Payment service fails → inconsistent state
- Two-phase commit (2PC) is expensive and still can fail (blocking during network partition)

### Trade-off: Eventual Consistency
- Services communicate via events/queues instead of transactions
- Compensating transactions needed to undo partial operations
- Example: Order created → Payment deducted → Email sent. If email fails, must reverse order or retry indefinitely

### Schema Evolution Risk
- Schema change in Service A doesn't affect Service B
- Must maintain backward compatibility longer (multiple service versions in prod simultaneously)
- Contract testing becomes critical

## When Each Wins

| Scenario | Winner | Why |
|----------|--------|-----|
| **Multi-step operations with rollback requirement** | Monolith | ACID guarantees eliminate coordination complexity |
| **High data coupling** (e.g., accounting ledgers) | Monolith | Consistency requirements justify single domain |
| **Independent failure domains** | Microservices | Payment failure shouldn't crash orders |
| **Different consistency levels per operation** | Microservices | Can use strong consistency for critical paths, eventual for non-critical |
| **Team scaling (10+ teams)** | Microservices | Isolated databases prevent team conflicts |
| **Strict compliance** (PCI, HIPAA) | Monolith initially | Easier to audit one database; microservices require per-service controls |

## Practical Implementation Patterns

### Monolith Pattern: Staged Rollout
```
Purchase Order → (atomic) debit + record → commit
If any step fails, entire transaction rolls back
Client sees success/failure atomically
```

### Microservices Pattern: Saga
```
OrderService.create() → emits "order.created"
PaymentService listens → debit account → emits "payment.completed"
InventoryService listens → reserve stock → emits "inventory.reserved"
If payment fails → emit "payment.failed" → OrderService reverts
```

## Real-World Trade-off Examples

1. **Netflix (Microservices)**: Accepts eventual consistency for recommendation engine. Worst case: outdated recommendations. Worth it for independent scaling.

2. **Banks (Monolith-first)**: Core ledger remains monolithic. Microservices only for non-critical (reports, UI). ACID consistency is non-negotiable.

3. **Stripe (Hybrid)**: Ledger is monolith + replicated. Services for card processing, webhooks use async events. Critical paths get consistency, high-traffic paths get speed.

## Cost of Being Wrong

- **Chose monolith, need 50+ independent services**: Rewrites required, painful migration
- **Chose microservices, needed ACID consistency**: Saga patterns become complex, bugs increase, deadlines slip

## Decision Checkpoint

Ask **before** architecting:
1. What operations MUST be atomic across domains?
2. What failures are acceptable (degraded service vs data corruption)?
3. Will we have 3+ independent teams? 5+?
4. What's the blast radius of data inconsistency?
