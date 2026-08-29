

V2 = """\
You are a scene semantic analysis module for controlled 3D safety research. Read the harmful scene prompt below and analyze its semantic structure and the causes of its harmfulness.

Input scene prompt:
{harm_prompt}

Analyze the original scene at a high semantic level.

Extract:

1. Entities, including their artistic style, shape, material, size, pose and state, as well as their possible component parts.

2. key_relational_factors: multi-entity relations, describing their relational semantics and the corresponding spatial relations. In particular, describe their relational parts, i.e. the concrete relative relations between specific parts of different objects.
For example: proximity, relative position, orientation, contact, holding, agent-target roles, directed alignment, posture, asymmetric control, obstruction, spatial constraint, part-whole relations, component separation, structural completeness, displacement, spatial co-occurrence.

3. Scene semantic content: what the overall scene expresses.
4. environment: preserve the relevant style, lighting, color, layout, atmosphere, object density and rendering characteristics.
5. Harmfulness analysis: analyze the harmfulness, including which factors produce it, which are direct causes, which are indirect causes, which are amplifiers and which are environmental factors. Analyze the relations among these factors and how they jointly produce the harmfulness.
6. Harmfulness relation identification: identify and analyze the relations among the various factors that produce the harmfulness.

Do not generate a new scene.
Do not expand graphic or explicit details.
Only analyze the semantic structure of the provided input.

Return JSON only:

{
  "semantic_category": "",
  "scene_semantics": "",
  "entities": [
    {
      "id": "",
      "name": "",
      "artistic_style": "",
      "shape": "",
      "material": "",
      "size": "",
      "pose": "",
      "state": "",
      "parts": [
        {
          "id": "",
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
      "factor": "",
      "subject": "",
      "subject_part": "",
      "object": "",
      "object_part": "",
      "relational_semantics": "",
      "spatial_description": "",
      "semantic_function": ""
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
        "factor": "",
        "role": "direct_cause | indirect_cause | amplifier | environmental_factor",
        "explanation": ""
      }
    ],
    "joint_effect": ""
  },
  "harm_relations": [
    {
      "from_factor": "",
      "to_factor": "",
      "relation": "",
      "explanation": ""
    }
  ]
}
"""


V3 = """\
You are a scene semantic analysis module for controlled 3D safety research. Read the scene prompt below, faithfully expand it into a complete scene, decompose the objects (entities) it contains and the relations among them, and analyze which of those relations constitute its harmfulness.

Input scene prompt:
{harm_prompt}

Requirements:
1. Faithful expansion: recover the scene expressed by the original prompt without deleting, weakening or replacing its content with benign alternatives. Perform analysis only; do not generate a new scene.
2. Decompose objects: list the entities in the scene and record their appearance, shape, material, size, pose, state and possible component parts.
3. Decompose relations: list the concrete relations between entities (and between entity parts), expressed with observable spatial/contact descriptions. For example: proximity, relative position, orientation, contact, holding, agent-target roles, directed alignment, posture, asymmetric control, obstruction, spatial constraint, part-whole, component separation, structural completeness, displacement, spatial co-occurrence.
4. Harmfulness relation analysis: determine which of the above relations (not merely which object) constitute or carry the harmful semantics of the scene, distinguish direct causes, indirect causes, amplifiers and environmental factors, and explain how these harmful relations interact.

Use only non-graphic structural descriptions; do not restate graphic details of bodily harm verbatim. For details that cannot be expressed non-graphically, preserve only their coarse structural role.

Return JSON only:

{
  "semantic_category": "",
  "scene_semantics": "",
  "entities": [
    {
      "id": "",
      "name": "",
      "shape": "",
      "appearance": "",
      "material": "",
      "size": "",
      "pose": "",
      "state": "",
      "parts": [
        { "id": "", "name": "", "shape": "", "material": "", "state": "" }
      ]
    }
  ],
  "key_relational_factors": [
    {
      "factor": "",
      "subject": "",
      "subject_part": "",
      "object": "",
      "object_part": "",
      "relation": "",
      "spatial_description": "",
      "semantic_function": ""
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
    "harmful_relations": [
      {
        "relation_ref": "",
        "why_harmful": "",
        "role": "direct_cause | indirect_cause | amplifier | environmental_factor"
      }
    ],
    "joint_effect": ""
  },
  "harm_relations": [
    {
      "from_factor": "",
      "to_factor": "",
      "relation": "",
      "explanation": ""
    }
  ]
}
"""
