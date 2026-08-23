import matplotlib.pyplot as plt

def plot_routes(routes, vessel_position=None, origin=None, destination=None, recommended_index=None):
    """
    Plot candidate routes on a lat/lon map.
    routes: list of Route objects
    vessel_position: (lat, lon) tuple - current ship position
    origin / destination: (lat, lon) tuples
    recommended_index: index of the recommended route (highlighted differently)
    """
    plt.figure(figsize=(8, 6))

    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    for i, route in enumerate(routes):
        lats = [wp.lat for wp in route.waypoints]
        lons = [wp.lon for wp in route.waypoints]

        if recommended_index is not None and i == recommended_index:
            plt.plot(lons, lats, color='red', linewidth=3, marker='o',
                      label=f'Route {i+1} (Recommended)', zorder=5)
        else:
            plt.plot(lons, lats, color=colors[i % len(colors)], linewidth=1.5,
                      linestyle='--', marker='o', alpha=0.7, label=f'Route {i+1}')

    if origin:
        plt.scatter(origin[1], origin[0], color='black', marker='s', s=100, label='Origin', zorder=6)
    if destination:
        plt.scatter(destination[1], destination[0], color='black', marker='*', s=200, label='Destination', zorder=6)
    if vessel_position:
        plt.scatter(vessel_position[1], vessel_position[0], color='cyan', marker='^',
                     s=150, edgecolor='black', label='Current Ship Position', zorder=7)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('TwinRoute-M: Candidate Voyage Routes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()