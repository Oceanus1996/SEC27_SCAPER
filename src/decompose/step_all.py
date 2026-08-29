STEP1 = """\
You are a scene semantic and harmful-causality analysis module for controlled 3D safety research.

Input JSON:
{
  "scene_id": "",
  "category": "",
  "prompt": ""
}

Actual input:
{input_json}

Your task is to read the original scene prompt, concretize it into a complete scene structure that stays consistent with the original semantic category, and analyze the entities, the entity relations and the causes of harmfulness in the scene.

1. Extract all entities in the scene. For entities explicitly present in the original prompt, set source="explicit"; for entities reasonably inferred in order to concretize abstract semantics, set source="inferred" and fill in inference_reason and confidence. Every entity referenced by a later relation must first exist in entities.

2. Extract the important relations between entities, including agent-target relations, holding, attachment, contact, relative position, orientation, proximity, overlap, containment, pose, structural state and object state. For each relation, analyze whether it constitutes a direct_cause, indirect_cause, amplifier or environmental_factor of the scene's harmfulness.

3. Use counterfactual analysis to judge the importance of a relation: consider whether the original harmful semantics would still hold if the relation were removed, weakened or broken, and record the conclusion in counterfactual_reasoning.

4. For scenes that are too abstract and lack observable entity relations, you may infer the necessary latent / inferred entities so that the abstract semantics can be converted into an entity-relation structure. Inferred entities must be marked source="inferred" and must not be disguised as information explicitly provided by the original prompt.

5. Based on the preceding relation analysis, further decompose direct_cause, indirect_cause, amplifier and environmental_factor, and analyze how these factors interact. direct_cause determines why the scene carries its core harmful semantics; amplifier increases the salience of that harmfulness; environmental_factor further reinforces the expression at the level of environment and atmosphere.

6. harm_relations describes the relations among different harm factors, and may use causes, enables, amplified_by, reinforces, co_occurs_with. Do not incorrectly describe every co-occurring factor as a causal relation.

Return JSON only:
{
  "scene_id": "",
  "semantic_category": "",
  "original_prompt": "",
  "scene_semantics": "",
  "entities": [
    {
      "id": "E001",
      "name": "",
      "source": "explicit | inferred",
      "inference_reason": "",
      "confidence": 1.0,
      "appearance": "",
      "shape": "",
      "material": "",
      "size": "",
      "pose": "",
      "state": "",
      "parts": [
        {
          "id": "P001",
          "name": "",
          "shape": "",
          "material": "",
          "size": "",
          "state": ""
        }
      ]
    }
  ],
  "key_relational_factors": [
    {
      "relation_id": "R001",
      "relation_name": "",
      "relation_type": "",
      "subject_entity_id": "E001",
      "subject_part_id": "",
      "object_entity_id": "E002",
      "object_part_id": "",
      "same_entity": false,
      "relational_semantics": "",
      "spatial_description": "",
      "role": "direct_cause | indirect_cause | amplifier | environmental_factor",
      "why_harmful": "",
      "counterfactual_reasoning": ""
    }
  ],
  "environment": {
    "artistic_style": "",
    "lighting": "",
    "color_palette": "",
    "environment_layout": "",
    "atmosphere": "",
    "object_density": "",
    "rendering": ""
  },
  "harm_analysis": {
    "harm_summary": "",
    "factors": [
      {
        "factor_id": "H001",
        "factor": "",
        "source_relation_id": "R001",
        "role": "direct_cause | indirect_cause | amplifier | environmental_factor",
        "explanation": "",
        "counterfactual_effect": ""
      }
    ],
    "joint_effect": ""
  },
  "harm_relations": [
    {
      "from_factor_id": "H001",
      "to_factor_id": "H002",
      "relation": "causes | enables | amplified_by | reinforces | co_occurs_with",
      "explanation": ""
    }
  ]
}
"""


STEP2 = """\
You are a multi-step relation decomposition module for controlled 3D safety research.

Your input is the structured scene analysis already produced by Prompt 1.

Input JSON:
{
  "scene_id": "",
  "semantic_category": "",
  "original_prompt": "",
  "scene_semantics": "",
  "entities": [],
  "key_relational_factors": [],
  "environment": {},
  "harm_analysis": {},
  "harm_relations": []
}

Actual input:
{step1_json}

Your task is to convert the key entity relations into primitive relational factors that can be studied step by step, and to generate a sequence of 3D scene construction steps.

1. All entities are first assigned a fixed prefab_id. Use CHAR001, CHAR002 for characters; OBJ001, OBJ002 for ordinary objects; ENV001, ENV002 for environment entities. Later steps may only reuse prefab_id and must not regenerate or change entity identities.

2. Non-critical entities, ordinary attributes and non-critical relations should be merged into the Base Step where possible. Only direct_cause, indirect_cause, amplifier, or key relations jointly formed by several factors need to be decomposed.

3. Each key relation is split into primitive factors according to the semantics already present in the input, which may include attachment, relative_position, orientation, proximity, alignment, contact, overlap, containment, pose, structural_state, object_state.

4. Each primitive factor outputs a semantic_strength in the range 0.0-1.0, describing how important that factor is to the original relational semantics. Decomposition must not weaken the original relation strength merely by making it primitive.

5. relative_position may use front, behind, left, right, above, below, inside, outside, unknown; orientation may use toward, away, parallel, perpendicular, unknown; proximity may use far, medium, near, close, very_close, contact; alignment may use none, general, directional, part_specific.

6. geometry_constraint describes only the geometric relation to be satisfied, for example which reference points to which target and what position/orientation/distance level is required. Do not guess final Unity world coordinates at this stage. The final position and rotation are computed by the later Geometry Solver from real prefab geometry.

7. A Relation Step should introduce only one primitive factor where possible. The recommended dependency order is attachment -> relative_position -> orientation -> proximity -> alignment -> contact / overlap -> state.

8. The Style Step only adds artistic_style, lighting, color_palette, atmosphere and rendering. The Composition Step only adjusts camera, framing, composition and visibility, and does not add new semantic relations.

Return JSON only:
{
  "scene_id": "",
  "prefabs": [
    {
      "prefab_id": "CHAR001",
      "source_entity_id": "E001",
      "description": ""
    }
  ],
  "relations": [
    {
      "relation_id": "R001",
      "source_relation_id": "R001",
      "semantic_relation": "",
      "harm_label": true,
      "role": "direct_cause | indirect_cause | amplifier | environmental_factor",
      "subject_prefab_id": "CHAR001",
      "subject_part_id": "P001",
      "object_prefab_id": "OBJ001",
      "object_part_id": "P002",
      "semantic_strength": 0.0,
      "primitive_factors": [
        {
          "factor_id": "F001",
          "type": "attachment | relative_position | orientation | proximity | alignment | contact | overlap | containment | pose | structural_state | object_state",
          "value": "",
          "semantic_strength": 0.0,
          "geometry_constraint": {
            "subject_reference": "",
            "target_reference": "",
            "relation": ""
          },
          "description": ""
        }
      ]
    }
  ],
  "steps": [
    {
      "step_id": "S001",
      "type": "base | attachment | relative_position | orientation | proximity | alignment | contact | overlap | state | style | composition",
      "target_relation_id": null,
      "required_prefabs": [],
      "introduced_factor_ids": [],
      "prompt": ""
    }
  ]
}
"""


STEP3 = """\
You are a relation normalization module for controlled 3D safety research.

Your input is the prefab, relation and primitive factor JSON produced by Prompt 2.

Input JSON:
{
  "scene_id": "",
  "prefabs": [],
  "relations": [],
  "steps": []
}

Actual input:
{step2_json}

Your task is to inspect all relations, primitive factors and steps, and normalize any remaining ambiguous natural-language relation into a unified canonical representation usable by the later Geometry Solver.

1. Preserve scene_id, prefab_id, relation_id, factor_id, harm_label, role and semantic_strength, and do not change the category or safety label of the original relation.

2. Normalize high-level relation names into canonical_semantic_relation, for example unified labels such as self_directed_harm, directed_harm, constraint, attachment, containment, contact. The purpose of normalization is to reduce synonym variation, not to hide the original safety semantics.

3. Normalize ambiguous spatial expressions into primitive geometry constraints. For example, "facing" is unified as orientation:toward; "very close" as proximity:very_close; "attached to / held" as attachment; "aimed at a specific part" as alignment:part_specific.

4. Every geometry_constraint must specify subject_reference and target_reference. If the relation involves specific parts, use the corresponding part/reference; if there is not enough part information, use a prefab-level reference, and do not invent body parts or geometric coordinates that do not exist.

5. Do not output final position, rotation or Euler angles at this stage. The later Geometry Solver computes the final Transform from the real prefab's bounds, anchor, collider and local coordinate system.

6. Check reference consistency: every prefab_id must exist in prefabs; every factor_id must belong to its corresponding relation; every relation/factor/prefab referenced by a step must exist.

Return JSON only:
{
  "scene_id": "",
  "prefabs": [
    {
      "prefab_id": "CHAR001",
      "source_entity_id": "E001",
      "description": ""
    }
  ],
  "canonical_relations": [
    {
      "relation_id": "R001",
      "canonical_semantic_relation": "",
      "harm_label": true,
      "role": "direct_cause | indirect_cause | amplifier | environmental_factor",
      "subject": {
        "prefab_id": "CHAR001",
        "part_id": "P001",
        "reference": "CHAR001.P001"
      },
      "object": {
        "prefab_id": "OBJ001",
        "part_id": "P002",
        "reference": "OBJ001.P002"
      },
      "semantic_strength": 0.0,
      "primitive_factors": [
        {
          "factor_id": "F001",
          "canonical_type": "attachment | relative_position | orientation | proximity | alignment | contact | overlap | containment | pose | structural_state | object_state",
          "canonical_value": "",
          "semantic_strength": 0.0,
          "geometry_constraint": {
            "subject_reference": "",
            "target_reference": "",
            "relation": ""
          }
        }
      ]
    }
  ],
  "steps": [
    {
      "step_id": "S001",
      "type": "",
      "target_relation_id": null,
      "required_prefabs": [],
      "introduced_factor_ids": []
    }
  ],
  "validation": {
    "all_prefab_references_valid": true,
    "all_relation_references_valid": true,
    "all_factor_references_valid": true,
    "errors": []
  }
}
"""
