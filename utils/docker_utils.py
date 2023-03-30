import docker


def get_containers_by_author(author: str) -> list:
    client = docker.from_env()
    images = client.images.list()
    container_names = []
    for img in images:
        if img.tags:    # some tags are empty
            if img.tags[0].startswith(author):
                container_name = img.tags[0].split('/')[1]
                container_name = container_name.split('-')[0]
                container_names.append(container_name)
    return container_names
