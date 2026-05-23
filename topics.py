class Topic:
    def __init__(self, topic_id, title, description, narrative):
        self.topic_id = topic_id
        self.title = title
        self.description = description
        self.narrative = narrative


def load_topics(path):
    """
    Function that parses a topics file and returns topic objects.\n
    Parameters: `path` - path of topics file to parse\n
    Return value: list of `Topic` objects
    """
    # List to store all parsed Topic objects
    topics = []

    # Temporary variables to store the current topic while reading the file
    current_id = None
    current_title = None
    current_description = []
    current_narrative = []

    # Keeps track of whether we are currently reading description or narrative lines
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove extra spaces and new line characters from each line
            stripped = line.strip()

            if stripped.startswith("<num>"):
                # Extract topic ID from the <num> line, e.g. "R101"
                parts = stripped.replace("<num>", "").replace("Number:", "").strip().split()
                current_id = parts[-1]

            elif stripped.startswith("<title>"):
                # Extract topic title from the <title> line
                current_title = stripped.replace("<title>", "").strip()
                current_section = None

            elif stripped.startswith("<desc>"):
                # The non-empty lines after this line belong to the topic description
                current_section = "description"

            elif stripped.startswith("<narr>"):
                # The non-empty lines after this line belong to the topic narrative
                current_section = "narrative"

            elif stripped.startswith("</Topic>"):
                # End of a topic block, so create a Topic object using the collected values
                if current_id:
                    topics.append(Topic(
                        current_id,
                        current_title,
                        " ".join(current_description).strip(),
                        " ".join(current_narrative).strip(),
                    ))

                # Reset temporary variables before reading the next topic
                current_id = None
                current_title = None
                current_description = []
                current_narrative = []
                current_section = None

            elif current_section == "description" and stripped:
                # Add description lines until the narrative section starts
                current_description.append(stripped)

            elif current_section == "narrative" and stripped:
                # Add narrative lines until the topic block ends
                current_narrative.append(stripped)

    return topics


def main():
    topics = load_topics("Topics.txt")

    for topic in topics:
        print(f"Topic ID: {topic.topic_id}")
        print(f"Title: {topic.title}")
        print(f"Description: {topic.description}")
        print(f"Narrative: {topic.narrative}")
        print()


if __name__ == "__main__":
    main()
