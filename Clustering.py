import math
import pygame
import numpy as np
from random import randint, uniform
from fcmeans import FCM
from sklearn.cluster import KMeans

BACKGROUND = (214, 214, 214)
BLACK = (0, 0, 0)
BACKGROUND_PANEL = (249, 255, 230)
WHITE = (255, 255, 255)

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (147, 153, 35)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)
COLORS = [RED,GREEN,BLUE,YELLOW,PURPLE,SKY,ORANGE,GRAPE,GRASS]

def draw_layout():
    global screen, text_plus, text_minus, num_cluster, text_run, text_random, text_algorithm, text_reset, mouse_x, mouse_y, algorithm
    screen.fill(BACKGROUND)

    # Draw panel
    pygame.draw.rect(screen, BLACK, (50, 50, 700, 500))
    pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 690, 490))

    # Algorithm button
    pygame.draw.rect(screen, BLACK, (850, 50, 150, 50))
    screen.blit(text_algorithm, (855, 50))

    # K button +
    pygame.draw.rect(screen, BLACK, (850, 150, 50, 50))
    screen.blit(text_plus, (865, 150))

    # K button -
    pygame.draw.rect(screen, BLACK, (950, 150, 50, 50))
    screen.blit(text_minus, (970, 150))

    # K value
    if algorithm == "K-Means":
        text_k = create_text_render("K = "+str(num_cluster), 40, BLACK)
        screen.blit(text_k, (1050, 150))
    elif algorithm == "Fuzzy":
        text_k = create_text_render("C = "+str(num_cluster), 40, BLACK)
        screen.blit(text_k, (1050, 150))
        
    # Random button
    pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
    screen.blit(text_random, (865, 250))

    # Method button
    pygame.draw.rect(screen, BLACK, (850, 350, 150, 50))
    screen.blit(text_method, (865, 350))

    # Run button
    pygame.draw.rect(screen, BLACK, (850, 450, 150, 50))
    screen.blit(text_run, (895, 450))

    # Reset button
    pygame.draw.rect(screen, BLACK, (850, 550, 150, 50))
    screen.blit(text_reset, (865, 550))
    
    # Draw mouse position when mouse in panel
    if 55 <= mouse_x <= 745 and 55 <= mouse_y <= 545:
        text_mouse = create_text_render("("+str(mouse_x-55)+","+str(mouse_y-55)+")", 20, BLACK)
        screen.blit(text_mouse, (mouse_x+5, mouse_y+5))

def create_text_render(string, font_size, color):
    font = pygame.font.SysFont('sans', font_size)
    return font.render(string, True, color)

def create_points():
    global mouse_x, mouse_y, points, labels, clusters, membership_matrix
    # Create points
    if 55 <= mouse_x <= 745 and 55 <= mouse_y <= 545:
        labels = []
        clusters = []
        membership_matrix = []
        point = [mouse_x-55, mouse_y-55]
        points.append(point)

def choose_algorithm():
    global num_algo_click, algorithm, text_algorithm, labels, clusters
    if 850 < mouse_x < 1000 and 50 < mouse_y < 100:
        labels = []
        clusters = []
        num_algo_click += 1
        if num_algo_click % 2 == 1:
            algorithm = "K-Means"
        else:
            algorithm = "Fuzzy"
        text_algorithm = create_text_render(algorithm, 40, WHITE)

def choose_number_cluster():
    global mouse_x, mouse_y, num_cluster
    # Plus button
    if 850 < mouse_x < 900 and 150 < mouse_y < 200:
        if num_cluster < 8:
            num_cluster += 1

    # Minus button
    if 950 < mouse_x < 1000 and 150 < mouse_y < 200:
        if num_cluster > 0:
            num_cluster -= 1

def generate_probabilities():
    global num_cluster
    if num_cluster < 2:
        raise ValueError("Number of objects must be at least 2.")
    
    # Generate random probabilities for each object
    probabilities = [uniform(0,1) for _ in range(num_cluster)]
    
    # Normalize probabilities to ensure they sum to 1
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]
    
    return probabilities

def get_random():
    global num_cluster, algorithm, clusters, labels, membership_matrix, points
    # Random Button
    if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
        if algorithm == "K-Means":
            clusters = []
            labels = []
            for i in range(num_cluster):
                random_point = [randint(5, 685), randint(5,485)]
                clusters.append(random_point)
        elif algorithm == "Fuzzy":
            labels = []
            clusters = []
            membership_matrix = []
            for p in points:
                probability = generate_probabilities()
                labels.append(probability.index(max(probability)))
                membership_matrix.append(probability)

def choose_method():
    global num_method_click, method, text_method
    # Run Button
    if 850 < mouse_x < 1000 and 350 < mouse_y < 400:
        num_method_click += 1
        if num_method_click % 2 == 1:
            method = "Scratch"
        else:
            method = "Library"
        text_method = create_text_render(method, 40, WHITE)
        
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
def run_kmeans_scratch():
    global labels, points, clusters, num_cluster
    labels = []
    # Assign point to closest clusters
    for p in points:
        distance_to_cluster = []
        for c in clusters:
            distance_to_cluster.append(distance(p, c))
        min_distance = min(distance_to_cluster)
        label = distance_to_cluster.index(min_distance)
        labels.append(label)

    # Update clusters
    for k in range(num_cluster):
        sum_x = 0
        sum_y = 0
        count = 0
        for j in range(len(points)):
            if labels[j] == k:
                sum_x += points[j][0]
                sum_y += points[j][1]
                count+=1
        if count != 0:
            new_cluster_x = int(sum_x / count)
            new_cluster_y = int(sum_y / count)
            clusters[k] = [new_cluster_x, new_cluster_y]

def run_kmeans_library():
    global num_cluster, points, clusters, labels
    kmeans = KMeans(n_clusters=num_cluster).fit(points)
    labels = kmeans.predict(points).tolist()
    clusters = kmeans.cluster_centers_.tolist()

def run_fuzzy_scratch():
    global membership_matrix, num_cluster, labels, points, clusters

    if len(membership_matrix) < len(points):
        get_random()
    
    clusters = []
    labels = []
    # Find cluster
    for i in range(num_cluster):
        numerator_x = 0
        denominator = 0
        numerator_y = 0
        for j in range(len(points)):
            p = points[j]
            numerator_x += membership_matrix[j][i]**2 * p[0]
            numerator_y += membership_matrix[j][i]**2 * p[1]
            denominator += membership_matrix[j][i]**2
        clusters.append([numerator_x/denominator, numerator_y/denominator])

    # Calculate distance and update the mebership matrix
    for i in range(len(points)):
        point = points[i]
        distances = []

        for j in range(num_cluster):
            cluster = clusters[j]
            distances.append(distance(point, cluster))

        for j in range(num_cluster):
            cur_distance = distances[j]
            gamma = 0
            for k in range(num_cluster):
                gamma += cur_distance**2 / distances[k]**2
            gamma = (gamma ** (2/(num_cluster-1)))**(-1)
            membership_matrix[i][j] = gamma
            
        labels.append(membership_matrix[i].index(max(membership_matrix[i])))

def run_fuzzy_library():
    global num_cluster, points, clusters, labels
    points = np.array(points)
    fcm = FCM(n_clusters = num_cluster)
    fcm.fit(points)
    labels = fcm.predict(points).tolist()
    clusters = fcm.centers.tolist()
    points = points.tolist()

def run():
    global clusters, mouse_x, mouse_y, algorithm, method
    if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
        if algorithm == "K-Means":
            if method == "Scratch":
                if clusters == []:
                    return
                run_kmeans_scratch()
            elif method == "Library":
                run_kmeans_library()
        elif algorithm == "Fuzzy":
            if method == "Scratch":
                if membership_matrix == []:
                    return
                run_fuzzy_scratch()
            elif method == "Library":
                run_fuzzy_library()

def reset():
    global mouse_x, mouse_y, num_cluster, error, clusters, points, labels, membership_matrix
    if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
        num_cluster = 0
        error = 0
        clusters = []
        points = []
        labels = []
        membership_matrix = []

if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((1200,700))

    pygame.display.set_caption("Clustering Visualization")

    algorithm = "Algorithm"
    method = "Method"
    text_algorithm = create_text_render(algorithm, 40, WHITE)
    text_plus = create_text_render("+", 40, WHITE)
    text_minus = create_text_render("-", 40, WHITE)
    text_random = create_text_render("Random", 40, WHITE)
    text_method = create_text_render(method, 40, WHITE)
    text_run = create_text_render("Run", 40, WHITE)
    text_reset = create_text_render("Reset", 40, WHITE)

    running = True

    clock = pygame.time.Clock()

    num_algo_click = 0
    num_cluster = 0
    num_method_click = 0
    error = 0
    points = []
    clusters = []
    labels = []
    membership_matrix = []

    while running:
        clock.tick(60)
        mouse_x, mouse_y = pygame.mouse.get_pos()

        draw_layout()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                create_points()
                choose_algorithm()
                choose_number_cluster()
                get_random()
                choose_method()
                run()
                reset()

        for i in range(len(clusters)):
            pygame.draw.circle(screen, COLORS[i], (55+int(clusters[i][0]), 55+int(clusters[i][1])), 10)

        # Draw points
        for i in range(len(points)):
            pygame.draw.circle(screen, BLACK, (points[i][0]+55, points[i][1]+55), 6)
            if labels == []:
                pygame.draw.circle(screen, WHITE, (points[i][0]+55, points[i][1]+55), 5)
            else:
                pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0]+55, points[i][1]+55), 5)

        # Calculate and draw error
        error = 0
        if clusters != [] and labels != []:
            for i in range(len(points)):
                error += distance(points[i], clusters[labels[i]])

        # Error text
        text_error = create_text_render("Error = "+str(int(error)), 40, BLACK)
        if method != "Method":
            screen.blit(text_error, (1010, 350))

        pygame.display.flip()

    pygame.quit()