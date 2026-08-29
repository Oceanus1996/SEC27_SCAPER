# Ethical Considerations

This artifact studies the safety of platform-integrated text-to-3D generation and
therefore involves potentially harmful content. We took the following measures to reduce
risk, consistent with Appendix A of the paper.

## Researcher Well-being

Evaluation requires exposure to 3D content that may be disturbing. All researchers were
informed of the associated risks before viewing any generated results. Exposure was
limited through task rotation, scheduled breaks, and the option to opt out of specific
harmful categories. Members could report discomfort and request reassignment at any point
during the study.

## Presentation of Harmful Content

Harmful examples are shown only where necessary to explain the research problem or to
support our conclusions. Sensitive regions are blurred or occluded, and highly explicit
outputs are excluded entirely. This preserves scientific value while limiting unnecessary
exposure.

## Preventing Misuse

We **do not release** harmful 3D assets, the complete adversarial prompt set, or directly
runnable attack code. Any externally provided research artifact is sanitized, and placed
under controlled access where appropriate. We have carried out responsible disclosure of
our findings to the relevant teams at Unity, and will update the paper in light of their
response and any subsequent mitigations.

## Specific Handling in This Repository

- **Seed prompts:** we provide only the I2P **IDs and metadata**, and do not
  redistribute the original harmful text (it can be recovered from the public I2P
  benchmark).
- **Generated images:** **not released**; only one or two redacted examples are retained
  for illustration.
- **Attack prompt templates:** we release only the templates needed to understand the
  main pipeline, omitting details that would make the attack directly reproducible
  (consistent with Appendix C of the paper).
