# MASTER SERVICES AGREEMENT

**Between:** Meridian Technologies Inc. ("Client")
**And:** Atlas Cloud Solutions LLC ("Provider")
**Effective Date:** January 15, 2026
**Agreement Number:** MSA-2026-0472

---

## 1. SCOPE OF SERVICES

Provider shall deliver the following cloud infrastructure and managed services
to Client for the duration of this Agreement:

### 1.1 Core Infrastructure
- Dedicated Kubernetes cluster (48 nodes, 192 vCPUs, 768 GB RAM)
- Multi-region deployment across US-East (Virginia), US-West (Oregon), and
  EU-West (Frankfurt)
- Guaranteed 99.95% uptime SLA measured monthly
- Automated failover with RPO < 15 minutes and RTO < 1 hour

### 1.2 Managed Database Services
- PostgreSQL 16 managed instances (primary + 2 read replicas per region)
- Automated daily backups with 90-day retention
- Point-in-time recovery capability
- Maximum storage allocation: 10 TB per region

### 1.3 Security & Compliance
- SOC 2 Type II certified infrastructure
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- Quarterly penetration testing by independent third party
- GDPR-compliant data processing for EU-region workloads

### 1.4 Support
- 24/7 Tier 1 support with 15-minute initial response SLA
- Dedicated Technical Account Manager (TAM)
- Monthly architecture review sessions
- Quarterly business reviews with executive sponsors

---

## 2. FINANCIAL TERMS

### 2.1 Total Contract Value
The total value of this Agreement is **$2,847,000** over the 36-month term,
structured as follows:

| Component | Monthly | Annual | 3-Year Total |
|-----------|---------|--------|--------------|
| Core Infrastructure | $48,500 | $582,000 | $1,746,000 |
| Managed Databases | $18,200 | $218,400 | $655,200 |
| Security & Compliance | $6,800 | $81,600 | $244,800 |
| Support & TAM | $5,583 | $67,000 | $201,000 |
| **Total** | **$79,083** | **$949,000** | **$2,847,000** |

### 2.2 Payment Terms
- Invoices issued on the 1st of each month
- Payment due within **Net 45** days of invoice date
- Wire transfer to Provider's designated account
- All amounts in USD

### 2.3 Late Payment
- Interest accrues at **1.5% per month** on overdue balances
- Compounded monthly from the due date
- Provider may suspend non-critical services after 60 days overdue
- Provider may terminate after 90 days overdue with 15 days written notice

### 2.4 Annual Escalation
- Rates increase by **3.5% annually** on each anniversary of the Effective Date
- Client will receive updated rate schedule 60 days before escalation
- Escalation applies to all service components uniformly

---

## 3. TERM AND RENEWAL

### 3.1 Initial Term
This Agreement has an initial term of **36 months** commencing on the
Effective Date (January 15, 2026) and expiring on **January 14, 2029**.

### 3.2 Renewal
- Automatic renewal for successive 12-month periods unless either party
  provides written notice of non-renewal at least **90 days** before expiration
- Renewal terms are subject to updated pricing (not to exceed 5% above
  the then-current rates)

### 3.3 Early Termination
- Either party may terminate for material breach with 30 days written notice
  and opportunity to cure
- Client may terminate for convenience with **180 days** written notice and
  payment of an early termination fee equal to **6 months** of the
  then-current monthly rate ($474,500 at initial rates)
- Provider may terminate immediately if Client fails to pay for 90+
  consecutive days

---

## 4. SERVICE LEVEL AGREEMENTS

### 4.1 Availability SLA
- Monthly uptime target: **99.95%**
- Measurement: (Total minutes - Downtime minutes) / Total minutes × 100
- Excludes scheduled maintenance windows (max 4 hours/month, weekends only)

### 4.2 SLA Credits

| Monthly Uptime | Credit (% of Monthly Fee) |
|----------------|--------------------------|
| 99.95% - 99.90% | 0% (within tolerance) |
| 99.90% - 99.50% | 10% |
| 99.50% - 99.00% | 25% |
| Below 99.00% | 50% |

- Credits applied to the next monthly invoice
- Maximum credit in any month: 50% of that month's fees
- Client must request credits within 30 days of the incident

### 4.3 Performance Benchmarks
- API response time: P95 < 200ms, P99 < 500ms
- Database query latency: P95 < 50ms for indexed queries
- Storage IOPS: minimum 10,000 sustained IOPS per region
- Network throughput: minimum 10 Gbps between regions

---

## 5. DATA AND INTELLECTUAL PROPERTY

### 5.1 Data Ownership
- All Client data remains the sole property of Client
- Provider acquires no rights to Client data except as needed to perform services
- Upon termination, Provider shall export all Client data within 30 days

### 5.2 Data Residency
- EU customer data must remain within the EU-West (Frankfurt) region
- US customer data may be processed in any US region
- No Client data shall be transferred outside the specified regions without
  prior written consent

### 5.3 Data Retention Post-Termination
- Provider retains Client data for **60 days** after termination for export
- After 60 days, all Client data is permanently deleted
- Provider issues a certificate of destruction upon request

---

## 6. LIABILITY AND INDEMNIFICATION

### 6.1 Limitation of Liability
- Provider's total aggregate liability shall not exceed **12 months** of fees
  actually paid by Client (approximately $949,000 at initial rates)
- Neither party is liable for indirect, consequential, or punitive damages
- Data breach liability is uncapped for Provider negligence

### 6.2 Indemnification
- Provider indemnifies Client against third-party IP infringement claims
- Client indemnifies Provider against claims arising from Client's use of
  services in violation of applicable law
- Indemnifying party covers reasonable legal fees and settlement amounts

---

## 7. MILESTONES AND DELIVERABLES

### 7.1 Implementation Timeline

| Milestone | Target Date | Deliverable |
|-----------|------------|-------------|
| Kickoff | February 1, 2026 | Project plan and team assignments |
| Infrastructure Provisioning | March 1, 2026 | All regions operational |
| Database Migration | April 15, 2026 | Data migrated, validated |
| Security Audit | May 1, 2026 | Penetration test report |
| Go-Live | **June 15, 2026** | Production traffic cutover |
| Hypercare | July 15, 2026 | 30-day stability period complete |

### 7.2 Acceptance Criteria
- Each milestone requires written sign-off from Client's project lead
- Provider has 10 business days to remediate any deficiencies
- Go-Live milestone requires successful load test at 2x projected peak traffic

---

## 8. GOVERNING LAW AND DISPUTE RESOLUTION

### 8.1 Governing Law
This Agreement is governed by the laws of the **State of Delaware**, without
regard to conflict of law principles.

### 8.2 Dispute Resolution
- Good faith negotiation for 30 days
- If unresolved, binding arbitration under AAA Commercial Rules
- Arbitration venue: Wilmington, Delaware
- Prevailing party entitled to reasonable attorney's fees

---

## SIGNATURES

**Meridian Technologies Inc.**
Name: Sarah Chen, VP of Engineering
Date: January 10, 2026

**Atlas Cloud Solutions LLC**
Name: Marcus Webb, Chief Revenue Officer
Date: January 12, 2026
