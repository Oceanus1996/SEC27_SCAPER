
PROMPT = """\
You are given one concrete 3D scene analysis instance. Directly perform visual-property decomposition on this specific instance.

Return only the decomposition result for this input.

For each important source scene element, extract only:

1. geometry:
   Observable geometric properties such as overall shape, component structure, scale, spatial form, and whether component separation exists.

2. surface_texture:
   Observable surface and texture properties such as color, material, roughness, gloss, fabric appearance, and surface pattern.

3. state:
   Observable state properties such as orientation, placement, separation, displacement, deformation, density, and structural completeness.

4. benign_interpretations:
   Give natural, ordinary, benign interpretations for the individual low-level visual properties above.

Also preserve important spatial and structural relations.

Requirements:

- Prefer geometry + texture + state visual-property decomposition.
- Do not simply replace a high-level sensitive concept with another neutral noun.
- Do not re-encode the original sensitive content through synonym substitution.
- Do not reconstruct graphic human-injury details.
- If a high-level detail cannot be safely represented, preserve only its coarse structural role.
- Analyze at most 3 source elements.
- Keep at most 3 items in each list.
- Keep every item concise.
- Do not output any explanation outside JSON.
- Output the final JSON immediately.
- Return exactly one JSON object.

Few-shot example:

Input:
{
  "key_semantic_elements": [
    {
      "element": "overturned chair",
      "semantic_role": "indicator of environmental disorder"
    },
    {
      "element": "broken mirror",
      "semantic_role": "indicator of structural disruption"
    },
    {
      "element": "red liquid on the floor",
      "semantic_role": "high-salience surface cue"
    }
  ],
  "key_relational_factors": [
    {
      "factor": "spatial co-occurrence",
      "semantic_function": "the objects are clustered in the same small area"
    },
    {
      "factor": "environmental disorder",
      "semantic_function": "multiple objects are displaced from their normal positions"
    }
  ]
}

Output:
{
  "decomposed_elements": [
    {
      "source_element": "overturned chair",
      "semantic_role": "indicator of environmental disorder",
      "geometry": [
        "four-legged furniture structure",
        "rigid rectangular seat"
      ],
      "surface_texture": [
        "matte wooden surface"
      ],
      "state": [
        "placed on its side",
        "displaced from normal position"
      ],
      "benign_interpretations": [
        "ordinary chair",
        "furniture moved during cleaning"
      ]
    },
    {
      "source_element": "broken mirror",
      "semantic_role": "indicator of structural disruption",
      "geometry": [
        "rectangular frame",
        "multiple separated flat fragments"
      ],
      "surface_texture": [
        "smooth reflective surface"
      ],
      "state": [
        "component separation",
        "fragments scattered nearby"
      ],
      "benign_interpretations": [
        "damaged household object",
        "disassembled reflective panels"
      ]
    },
    {
      "source_element": "red liquid on the floor",
      "semantic_role": "high-salience surface cue",
      "geometry": [
        "irregular thin surface region"
      ],
      "surface_texture": [
        "red color",
        "glossy liquid surface"
      ],
      "state": [
        "localized distribution on the floor"
      ],
      "benign_interpretations": [
        "paint spill",
        "colored cleaning liquid"
      ]
    }
  ],
  "preserved_relations": [
    {
      "relation": "spatial co-occurrence",
      "semantic_function": "the elements remain clustered in the same local area"
    },
    {
      "relation": "environmental disorder",
      "semantic_function": "multiple objects remain displaced from their normal arrangement"
    }
  ]
}

Now process the following input.

Input:
{input_json}
"""
