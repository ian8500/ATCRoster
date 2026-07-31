# ATCRoster interface design system

ATCRoster uses a restrained operational interface intended for prolonged,
high-attention use. It should feel like mature institutional software, not a
marketing dashboard.

## Principles

1. **Information before decoration.** Layout, labels and state carry meaning;
   gradients, glow, animation and ornamental graphics do not.
2. **One hierarchy.** Page heading, section heading, field label and supporting
   text have consistent size and weight across every feature.
3. **One neutral system.** Graphite surfaces and grey borders form the
   interface. Muted aviation teal marks selection and primary action.
4. **Colour has a job.** Roster duty colours, warnings, errors and publication
   states retain their operational meaning. Do not use those colours merely for
   decoration.
5. **Compact, not cramped.** Controls suit dense roster work while maintaining
   readable labels, minimum touch targets and visible keyboard focus.
6. **Flat, accountable controls.** Avoid glass effects, heavy shadows, animated
   backgrounds, excessive rounding and competing button treatments.

## Core tokens

| Use | Value |
|---|---|
| Page background | `#0d1117` |
| Primary surface | `#151a21` |
| Elevated surface | `#1a2028` |
| Border | `#2a333e` |
| Primary text | `#e8edf2` |
| Muted text | `#9aa5b1` |
| Accent | `#68b7b2` |
| Primary action | `#356d69` |
| Standard radius | `6–10px` |

The font stack is the operating system's native UI family with a native
monospace stack for times, codes and technical identifiers. This avoids an
external font dependency and keeps rendering familiar on managed devices.

## Components

- Header: solid surface, simple `AR` identifier, flat navigation and restrained
  session context.
- Page introduction: bordered surface with a narrow accent edge, one heading
  and one concise explanation.
- Cards: flat surface, one-pixel border and minimal shadow.
- Buttons: neutral by default; filled teal only for the primary action; muted
  red only for destructive actions.
- Tables: flat header, sentence-case labels, subtle alternating rows and no
  decorative gradient.
- Forms: dark input surface, visible labels, consistent 38px minimum height and
  teal focus ring.
- Mobile administration: two-column tool grid, becoming one column only where
  content requires it; no hidden horizontal toolbar.

## Change rule

New component styling belongs in the final institutional-design section of
`static/styles.css` until legacy rules are progressively removed. Any new
colour, radius, shadow or font treatment needs a functional reason and must be
checked at desktop and 390px mobile width.
