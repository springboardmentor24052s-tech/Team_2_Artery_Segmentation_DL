
import cv2
import numpy as np
import networkx as nx


def build_skeleton_graph(centerline_mask):
    """
    Convert a one-pixel-wide skeleton into a graph.

    Each node represents a skeleton pixel (y, x).
    Redundant diagonal edges are suppressed when an
    orthogonal connection already exists.
    """
    if centerline_mask is None or centerline_mask.size == 0:
        raise ValueError("Centerline mask is empty.")

    if len(centerline_mask.shape) == 3:
        centerline_mask = cv2.cvtColor(
            centerline_mask, cv2.COLOR_BGR2GRAY
        )

    skeleton = centerline_mask > 0
    height, width = skeleton.shape

    graph = nx.Graph()
    pixels = np.argwhere(skeleton)

    for y, x in pixels:
        graph.add_node((int(y), int(x)))

    for y, x in pixels:
        y, x = int(y), int(x)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                if dy == 0 and dx == 0:
                    continue

                ny, nx_ = y + dy, x + dx

                if not (
                    0 <= ny < height
                    and 0 <= nx_ < width
                    and skeleton[ny, nx_]
                ):
                    continue

                # Avoid redundant diagonal connections.
                if dy != 0 and dx != 0:
                    if (
                        skeleton[y + dy, x]
                        or skeleton[y, x + dx]
                    ):
                        continue

                graph.add_edge(
                    (y, x),
                    (ny, nx_)
                )

    return graph


def extract_vessel_branches(centerline_mask):
    """
    Extract ordered skeleton paths between endpoints and junctions.

    Branch coordinates are returned as (y, x).
    """
    graph = build_skeleton_graph(centerline_mask)

    if graph.number_of_nodes() == 0:
        return {
            "skeleton_pixel_count": 0,
            "connected_component_count": 0,
            "endpoint_count": 0,
            "junction_pixel_count": 0,
            "junction_region_count": 0,
            "isolated_pixel_count": 0,
            "branch_count": 0,
            "branches": []
        }

    degrees = dict(graph.degree())

    endpoints = {
        node for node, degree in degrees.items()
        if degree == 1
    }

    junction_pixels = {
        node for node, degree in degrees.items()
        if degree >= 3
    }

    isolated_pixels = {
        node for node, degree in degrees.items()
        if degree == 0
    }

    # Group adjacent junction pixels.
    junction_graph = graph.subgraph(junction_pixels)

    junction_regions = list(
        nx.connected_components(junction_graph)
    )

    # Endpoints and junction regions are critical regions.
    critical_regions = [
        {endpoint} for endpoint in endpoints
    ] + junction_regions

    pixel_to_region = {}

    for region_id, region in enumerate(critical_regions):
        for pixel in region:
            pixel_to_region[pixel] = region_id

    visited_edges = set()
    branches = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    # Trace paths from every critical region.
    for region_id, region in enumerate(critical_regions):

        for start_pixel in region:

            for neighbour in graph.neighbors(start_pixel):

                if neighbour in region:
                    continue

                first_edge = edge_key(start_pixel, neighbour)

                if first_edge in visited_edges:
                    continue

                path = [start_pixel, neighbour]
                visited_edges.add(first_edge)

                previous = start_pixel
                current = neighbour

                while current not in pixel_to_region:

                    next_pixels = [
                        pixel
                        for pixel in graph.neighbors(current)
                        if pixel != previous
                    ]

                    if not next_pixels:
                        break

                    # A regular path pixel should have one
                    # forward neighbour.
                    next_pixel = next_pixels[0]

                    next_edge = edge_key(current, next_pixel)

                    if next_edge in visited_edges:
                        break

                    visited_edges.add(next_edge)
                    path.append(next_pixel)

                    previous = current
                    current = next_pixel

                # Record only paths reaching another critical region.
                if current in pixel_to_region:
                    branches.append({
                        "branch_id": len(branches) + 1,
                        "start_node": region_id,
                        "end_node": pixel_to_region[current],
                        "pixel_count": len(path),
                        "coordinates": [
                            [int(y), int(x)]
                            for y, x in path
                        ]
                    })

    return {
        "skeleton_pixel_count": graph.number_of_nodes(),
        "connected_component_count": nx.number_connected_components(graph),
        "endpoint_count": len(endpoints),
        "junction_pixel_count": len(junction_pixels),
        "junction_region_count": len(junction_regions),
        "isolated_pixel_count": len(isolated_pixels),
        "branch_count": len(branches),
        "branches": branches
    }
