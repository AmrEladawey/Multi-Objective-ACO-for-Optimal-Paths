if __name__ == "__main__":

    cities = generate_random_cities(5)
    world = Map(cities)
    path = [0,4,1,2,3,0]
    
    visualize_cities(world,path)