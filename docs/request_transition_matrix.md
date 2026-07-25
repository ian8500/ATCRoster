# Shift-request transition matrix

| Current | Action | Result | Conditions |
| --- | --- | --- | --- |
| Pending | Approve only | Approved | Unit Admin; audited |
| Pending | Reject | Rejected | Unit Admin; audited |
| Pending | Requester cancel | Cancelled | Owner, unlocked request |
| Pending | Admin cancel | Cancelled | Unit Admin; audited reason |
| Pending | Approve and apply | Fulfilled | Assignment, fulfilment fields, audit and notification commit together |
| Approved | Approve and apply | Fulfilled | Same atomic validation and commit |
| Approved | Reject | Rejected | Reason of at least 10 characters |
| Approved | Cancel | Cancelled | Explicit Unit Admin action and reason |
| Rejected | Any transition | Forbidden | Terminal |
| Cancelled | Any transition | Forbidden | Terminal |
| Fulfilled | Any transition | Forbidden | Terminal and idempotent |

`fulfilled` is not offered in the generic status selector. It can only result
from successful **Approve and apply** after tenant, shift, roster lock,
publication, fatigue and qualification validation. A permitted conflict
override requires Unit Admin or explicit override permission and an audited
reason. Any exception before commit rolls back both assignment and request.
