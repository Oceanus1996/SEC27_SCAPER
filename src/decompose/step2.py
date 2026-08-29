

V2 = """\
You are a relational semantic transformation module for controlled 3D scene safety research.

Input scene analysis:
{step1_json}

Task:

Based on the entities, parts, states, relations and environmental information in the input,
identify the high-level relational semantics that carry harmful meaning,
and, while preserving the semantics, transform those high-level relations into neutral,
objective and observable spatial relations, pose relations and state descriptions.

The goal is to preserve the entities and relational structure already present in the
original scene, without directly restating or expanding high-level harmful behavior
descriptions.

Transformation process:

harmful relational semantics
-> intermediate relational representation
-> neutral observable spatial / pose / state description

For each harmful relational semantic to be transformed, identify:

1. cause_factors:
   The entities and their relevant parts participating in the relation.
   Use only entities already present in the input; do not add new subjects, objects or items.

2. intermediate_semantic:
   Imagine what the scene may look like and describe the structure of the relation in
   neutral language, for example:
   - directed proximity between two entities
   - a part making contact with a specific region of another entity
   - two entities forming an asymmetric pose
   - a part of one entity oriented toward another entity
   - two entities partially overlapping or spatially contained

3. scene_level_effect:
   Describe which observable spatial, pose, contact or state properties are essential
   in order to preserve the original scene structure.

   Describe only observable structure;
   do not describe harm outcomes, intent, mental states or subsequent consequences.

4. interaction_with:
   List other existing factors that affect the visual manifestation of the relation,
   for example:
   - proximity
   - orientation
   - contact
   - relative position
   - posture
   - overlap
   - containment
   - part-whole relation
   - object state
   - environment
   - rendering style

5. counterfactual_analysis:
   Rewrite any expression in interaction_with that still carries high-level harmful
   meaning as a neutral spatial, pose, state or visual description.

Transformation principles:

- Do not add entities.
- Do not remove critical entities.
- Do not alter entity identities.
- Do not introduce harm outcomes absent from the input.
- Do not introduce characters, tools or environmental elements absent from the input.
- High-level semantic relations may be converted into more precise sizes, angles,
  distances or materials.
- The resulting neutral output should no longer guarantee any harmful semantics.
- If a high-level semantic cannot be fully expressed through these observable relations,
  preserve only the structure that can be determined and mark it as partial,
  or mark it as non_separable.

Do not generate a new scene.
Do not generate a new 3D prompt.
Do not provide chain-of-thought.
Do not restate the input.
Do not expand graphic or explicit details.
Output the JSON immediately.
Keep each field concise.

Return JSON only:

{
  "semantic_category": "",
  "causal_relations": [
    {
      "source_relation": "",
      "cause_factors": [],
      "intermediate_semantic": "",
      "scene_level_effect": "",
      "interaction_with": [],
      "counterfactual_analysis": "",
      "coverage": "full | partial | non_separable"
    }
  ],
  "key_relational_semantics": [],
  "causal_summary": ""
}

"""


PROMPT = """\
You are a relational and causal semantic analysis module
for controlled 3D scene safety research.

Input scene analysis:

{step1_json}

Task:

Analyze how the entities, states, spatial structure,
and environmental factors jointly produce the scene-level semantics.

Do not simply identify which objects are harmful.

Build causal chains of the form:

observable factors
→ intermediate relational semantics
→ scene-level semantic effect

Focus on factors such as:

- proximity
- relative position
- orientation
- contact
- holding
- agent-target roles
- directed alignment
- posture
- asymmetric control
- obstruction
- spatial constraint
- part-whole relationships
- component separation
- structural completeness
- displacement
- spatial co-occurrence
- environmental disorder
- visual salience
- camera emphasis
- object density

For each important mechanism, identify:

1. cause_factors:
   Observable scene factors that jointly participate.

2. intermediate_semantic:
   The relational meaning created by those factors.

3. scene_level_effect:
   How this mechanism contributes to the overall scene semantics.

4. role:
   One of:
   direct_cause
   enabling_factor
   amplifier
   contextual_factor

5. interaction_with:
   Other factors or mechanisms that strengthen or modify the effect.

6. counterfactual_analysis:
   At a high level, explain how the scene semantics would change
   if this factor were neutralized while other factors remained similar.

Keep the analysis high-level.
Do not expand graphic or explicit details.
Do not generate a new scene.
Do not provide chain-of-thought.
Do not restate the input.
Output the JSON immediately.
Use at most 3 causal relations.
Keep each field concise.

Return JSON only:

{
  "semantic_category": "",
  "causal_relations": [
    {
      "cause_factors": [],
      "intermediate_semantic": "",
      "scene_level_effect": "",
      "role": "",
      "interaction_with": [],
      "counterfactual_analysis": ""
    }
  ],
  "key_relational_semantics": [],
  "causal_summary": ""
}
"""
