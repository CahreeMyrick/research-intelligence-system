import networkx as nx


# ============================================================
# Knowledge Graph Setup
# ============================================================

G = nx.MultiDiGraph()


# ============================================================
# Helper Functions
# ============================================================

def add_entity(entity_id, entity_type, **properties):
    """
    Add an entity node to the knowledge graph.
    """
    G.add_node(
        entity_id,
        entity_type=entity_type,
        **properties,
    )


def add_relation(source, relation, target, **properties):
    """
    Add a directed relationship between two entities.
    """
    G.add_edge(
        source,
        target,
        relation=relation,
        **properties,
    )


# ============================================================
# Canonical Anatomy Entities
# ============================================================

add_entity(
    "RightPleuralSpace",
    "Anatomy",
    name="Right Pleural Space",
    aliases=["right pleural space"],
)

add_entity(
    "Heart",
    "Anatomy",
    name="Heart",
    aliases=["heart", "cardiac"],
)

add_entity(
    "RightLung",
    "Anatomy",
    name="Right Lung",
    aliases=["right lung"],
)

add_entity(
    "LeftLung",
    "Anatomy",
    name="Left Lung",
    aliases=["left lung"],
)


# ============================================================
# Canonical Radiographic Findings & Symptoms
# ============================================================

add_entity(
    "PleuralEffusion",
    "RadiographicFinding",
    name="Pleural Effusion",
    aliases=["pleural effusion", "pleural fluid"],
)

add_entity(
    "Cardiomegaly",
    "RadiographicFinding",
    name="Cardiomegaly",
    aliases=["cardiomegaly", "enlarged heart"],
)

add_entity(
    "PulmonaryEdema",
    "RadiographicFinding",
    name="Pulmonary Edema",
    aliases=["pulmonary edema", "fluid in lungs"],
)

add_entity(
    "Atelectasis",
    "RadiographicFinding",
    name="Atelectasis",
    aliases=["atelectasis", "partial lung collapse"],
)

add_entity(
    "Pneumothorax",
    "RadiographicFinding",
    name="Pneumothorax",
    aliases=["pneumothorax", "collapsed lung"],
)

add_entity(
    "ChestPain",
    "Symptom",
    name="Chest Pain",
    aliases=["chest pains", "chest pain", "anginal pain", "precordial pain"],
)

add_entity(
    "Dyspnea",
    "Symptom",
    name="Shortness of Breath",
    aliases=["dyspnea", "shortness of breath", "breathlessness"],
)


# ============================================================
# Activities and Triggers
# ============================================================

add_entity(
    "Running",
    "Activity",
    name="Running",
    aliases=["running", "jogging"],
)

add_entity(
    "Exercise",
    "Activity",
    name="Exercise",
    aliases=["exercise", "physical exercise"],
)

add_entity(
    "PhysicalExertion",
    "Activity",
    name="Physical Exertion",
    aliases=["physical exertion", "exertion", "strenuous activity", "physical activity"],
)


# ============================================================
# Clinical Conditions
# ============================================================

add_entity(
    "HeartFailure",
    "Disease",
    name="Heart Failure",
    aliases=["heart failure", "congestive heart failure", "chf"],
)

add_entity(
    "Pneumonia",
    "Disease",
    name="Pneumonia",
    aliases=["pneumonia", "lung infection"],
)

add_entity(
    "LungCollapse",
    "Disease",
    name="Lung Collapse",
    aliases=["lung collapse"],
)

add_entity(
    "Angina",
    "Condition",
    name="Angina Pectoris",
    aliases=["angina", "angina pectoris", "stable angina"],
)

add_entity(
    "CoronaryArteryDisease",
    "Disease",
    name="Coronary Artery Disease",
    aliases=["coronary artery disease", "cad", "coronary heart disease"],
)

add_entity(
    "MyocardialIschemia",
    "Disease",
    name="Myocardial Ischemia",
    aliases=["myocardial ischemia", "cardiac ischemia"],
)


# ============================================================
# Attributes
# ============================================================

add_entity(
    "Moderate",
    "Severity",
    name="Moderate",
)

add_entity(
    "Mild",
    "Severity",
    name="Mild",
)

add_entity(
    "Right",
    "Laterality",
    name="Right",
)

add_entity(
    "Left",
    "Laterality",
    name="Left",
)


# ============================================================
# General Medical Knowledge & Associations
# ============================================================

# Ontological relationships
add_relation("Running", "IS_A", "Exercise")
add_relation("Exercise", "IS_A", "PhysicalExertion")

# Cardiovascular & Exertion knowledge
add_relation("Angina", "HAS_SYMPTOM", "ChestPain")
add_relation("Angina", "TRIGGERED_BY", "PhysicalExertion")
add_relation("Angina", "ASSOCIATED_WITH", "CoronaryArteryDisease")
add_relation("CoronaryArteryDisease", "MAY_CAUSE", "Angina")
add_relation("CoronaryArteryDisease", "HAS_SYMPTOM", "ChestPain")
add_relation("CoronaryArteryDisease", "TRIGGERED_BY", "PhysicalExertion")
add_relation("MyocardialIschemia", "HAS_SYMPTOM", "ChestPain")
add_relation("MyocardialIschemia", "TRIGGERED_BY", "PhysicalExertion")
add_relation("HeartFailure", "HAS_SYMPTOM", "Dyspnea")
add_relation("HeartFailure", "TRIGGERED_BY", "PhysicalExertion")
add_relation("Heart", "ASSOCIATED_WITH", "Angina")
add_relation("Heart", "ASSOCIATED_WITH", "CoronaryArteryDisease")

# Radiographic finding associations
add_relation(
    "PleuralEffusion",
    "ASSOCIATED_WITH",
    "HeartFailure",
)

add_relation(
    "PleuralEffusion",
    "ASSOCIATED_WITH",
    "Pneumonia",
)

add_relation(
    "Cardiomegaly",
    "ASSOCIATED_WITH",
    "HeartFailure",
)

add_relation(
    "PulmonaryEdema",
    "ASSOCIATED_WITH",
    "HeartFailure",
)

add_relation(
    "Atelectasis",
    "ASSOCIATED_WITH",
    "Pneumonia",
)

add_relation(
    "Pneumothorax",
    "ASSOCIATED_WITH",
    "LungCollapse",
)


# ============================================================
# Example Chest X-Ray
# ============================================================

add_entity(
    "Image_123",
    "ChestXRay",
    filename="xray_123.png",
)


# ============================================================
# Findings Observed In This Image
# ============================================================

add_entity(
    "Finding_001",
    "FindingObservation",
    concept="PleuralEffusion",
    semantic_score=0.41,
    localization_score=0.88,
    bbox=[340, 410, 600, 720],
    source_model="BioMedCLIP+ChEX",
)

add_entity(
    "Finding_002",
    "FindingObservation",
    concept="Cardiomegaly",
    semantic_score=0.37,
    localization_score=0.81,
    bbox=[210, 260, 660, 650],
    source_model="BioMedCLIP+ChEX",
)


# ============================================================
# Image -> Finding Relations
# ============================================================

add_relation(
    "Image_123",
    "SHOWS_FINDING",
    "Finding_001",
)

add_relation(
    "Image_123",
    "SHOWS_FINDING",
    "Finding_002",
)


# ============================================================
# Finding Observation -> Canonical Concept
# ============================================================

add_relation(
    "Finding_001",
    "INSTANCE_OF",
    "PleuralEffusion",
)

add_relation(
    "Finding_002",
    "INSTANCE_OF",
    "Cardiomegaly",
)


# ============================================================
# Spatial Relations
# ============================================================

add_relation(
    "Finding_001",
    "LOCATED_IN",
    "RightPleuralSpace",
)

add_relation(
    "Finding_002",
    "LOCATED_IN",
    "Heart",
)


# ============================================================
# Attribute Relations
# ============================================================

add_relation(
    "Finding_001",
    "HAS_SEVERITY",
    "Moderate",
)

add_relation(
    "Finding_001",
    "HAS_LATERALITY",
    "Right",
)

add_relation(
    "Finding_002",
    "HAS_SEVERITY",
    "Mild",
)


# ============================================================
# Inspection Utilities
# ============================================================

def print_entities():
    print("\n=== ENTITIES ===\n")

    for node, properties in G.nodes(data=True):
        print(f"{node}")
        for key, value in properties.items():
            print(f"  {key}: {value}")
        print()


def print_relations():
    print("\n=== RELATIONSHIPS ===\n")

    for source, target, data in G.edges(data=True):
        relation = data.get("relation")

        print(
            f"{source}"
            f" --{relation}--> "
            f"{target}"
        )


# ============================================================
# Simple Graph Queries
# ============================================================

def associated_conditions(finding):
    """
    Return diseases associated with a canonical finding.
    """
    conditions = set()

    for _, target, data in G.out_edges(
        finding,
        data=True,
    ):
        if data.get("relation") == "ASSOCIATED_WITH":
            conditions.add(target)

    return conditions


def get_image_findings(image_id):
    """
    Return finding observation IDs for a chest X-ray.
    """
    findings = []

    for _, target, data in G.out_edges(
        image_id,
        data=True,
    ):
        if data.get("relation") == "SHOWS_FINDING":
            findings.append(target)

    return findings


def get_canonical_concept(finding_observation):
    """
    Resolve a finding observation to its canonical concept.
    """
    for _, target, data in G.out_edges(
        finding_observation,
        data=True,
    ):
        if data.get("relation") == "INSTANCE_OF":
            return target

    return None


def get_related_conditions_for_image(image_id):
    """
    Find diseases related to the findings observed in an image.
    """

    findings = get_image_findings(image_id)

    results = {}

    for finding_observation in findings:

        concept = get_canonical_concept(
            finding_observation
        )

        if concept is None:
            continue

        conditions = associated_conditions(
            concept
        )

        results[concept] = conditions

    return results


def find_shared_conditions(image_id):
    """
    Find diseases associated with multiple findings in the image.
    """

    finding_map = get_related_conditions_for_image(
        image_id
    )

    condition_counts = {}

    for finding, conditions in finding_map.items():

        for condition in conditions:

            condition_counts[condition] = (
                condition_counts.get(
                    condition,
                    0
                )
                + 1
            )

    shared = {
        condition: count
        for condition, count
        in condition_counts.items()
        if count > 1
    }

    return shared


# ============================================================
# Visualization
# ============================================================

def visualize_graph():
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.8,
    )

    plt.figure(
        figsize=(18, 12)
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2500,
    )

    # Draw labels
    labels = {}

    for node, properties in G.nodes(
        data=True
    ):

        display_name = properties.get(
            "name",
            node
        )

        labels[node] = display_name

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.06",
    )

    # Edge labels
    edge_labels = {}

    for source, target, key, data in G.edges(
        keys=True,
        data=True,
    ):
        relation = data.get(
            "relation",
            ""
        )

        edge_labels[
            (source, target)
        ] = relation

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
    )

    plt.title(
        "Chest X-Ray Knowledge Graph"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print_entities()

    print_relations()

    print(
        "\n=== IMAGE FINDINGS ===\n"
    )

    findings = get_image_findings(
        "Image_123"
    )

    print(findings)


    print(
        "\n=== RELATED CLINICAL CONDITIONS ===\n"
    )

    condition_map = (
        get_related_conditions_for_image(
            "Image_123"
        )
    )

    for finding, conditions in condition_map.items():

        print(
            f"{finding}: "
            f"{sorted(conditions)}"
        )


    print(
        "\n=== SHARED CLINICAL CONDITIONS ===\n"
    )

    shared = find_shared_conditions(
        "Image_123"
    )

    for condition, count in shared.items():

        name = G.nodes[
            condition
        ].get(
            "name",
            condition
        )

        print(
            f"{name} is associated with "
            f"{count} observed findings."
        )


    visualize_graph()


if __name__ == "__main__":
    main()
