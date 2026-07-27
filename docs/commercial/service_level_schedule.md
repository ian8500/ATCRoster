# Service level schedule — draft

This schedule must be reviewed against actual infrastructure and on-call
capability before signature.

## Service target

Proposed initial paid-service objective: **99.5% monthly availability**, measured
at ATCRoster's public application boundary. This is approximately 3 hours 39
minutes maximum unavailability in a 31-day month. Do not contract to this target
until external monitoring, resilient backups, paging and tested support exist.

Excluded from measurement:

- announced maintenance within the agreed window;
- customer systems, internet access or unsupported configuration;
- customer misuse or unauthorised changes;
- suspension required for security, law or non-payment;
- force majeure; and
- provider failure only to the extent the customer contract lawfully excludes
  it—provider dependency is not automatically ignored operationally.

## Maintenance

Give at least five business days' notice for planned material interruption
where practicable. Emergency security maintenance may occur without advance
notice; communicate promptly.

## Support

Priority definitions and response objectives are in the support policy. Unless
the order form purchases on-call coverage, support hours are UK business days
09:00–17:00 Europe/London.

## Recovery objectives

Proposed objectives, subject to demonstrated restore evidence:

- RPO: 24 hours initially; target 4 hours before larger commercial adoption.
- RTO: 8 hours initially.

These are disaster-recovery objectives, not guarantees that every incident is
resolved within the period.

## Service credits

Do not offer credits until pricing and liability are approved. A conventional
draft is 5% of the affected month's recurring fee below 99.5%, 10% below 99.0%
and 20% below 95.0%, capped at 20%, claimed within 30 days. Credits should be
the sole financial remedy for availability failure without excluding remedies
that law does not permit excluding.
