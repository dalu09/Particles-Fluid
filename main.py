import cv2
import numpy as np
import pygame
from scipy.spatial import cKDTree
from math import exp, radians, sqrt, tan
from pygame.locals import *
from OpenGL.GL import *

# Importación estándar de Mediapipe
import mediapipe as mp

# Vértices y aristas del cubo 3D
vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = (
    (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7),
    (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7)
)
FIST_EXTENSION_RATIO = 1.2
OPEN_EXTENSION_RATIO = 2.2
CLOSED_FIST_DISTANCE = 7.0
OPEN_HAND_DISTANCE = 3.5
FLUID_PARTICLE_COUNT = 500
FLUID_WALL_LIMIT = 0.88
FLUID_NEIGHBOR_RADIUS = 0.2
FLUID_PARTICLE_SPACING = 0.075

def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def create_fluid_particles():
    random = np.random.default_rng(7)
    positions = random.uniform(
        (-0.72, -0.72, -0.72), (0.72, 0.25, 0.72),
        (FLUID_PARTICLE_COUNT, 3)
    ).astype(np.float32)
    velocities = random.normal(0, 0.04, (FLUID_PARTICLE_COUNT, 3)).astype(np.float32)
    return positions, velocities

def step_fluid(positions, velocities, delta_time, rot_x, rot_y, angular_velocity):
    if delta_time <= 0:
        return

    pairs = cKDTree(positions).query_pairs(
        FLUID_NEIGHBOR_RADIUS, output_type="ndarray"
    )
    pressure = np.zeros_like(positions)
    velocity_sum = np.zeros_like(velocities)
    neighbor_counts = np.zeros((len(positions), 1), dtype=np.float32)
    if len(pairs):
        first, second = pairs.T
        pair_offsets = positions[first] - positions[second]
        pair_distances = np.linalg.norm(pair_offsets, axis=1)
        pair_directions = pair_offsets / np.maximum(pair_distances[:, None], 1e-6)
        overlap = np.maximum(FLUID_PARTICLE_SPACING - pair_distances, 0)
        pair_pressure = pair_directions * overlap[:, None] * 32
        np.add.at(pressure, first, pair_pressure)
        np.add.at(pressure, second, -pair_pressure)
        np.add.at(velocity_sum, first, velocities[second])
        np.add.at(velocity_sum, second, velocities[first])
        np.add.at(neighbor_counts[:, 0], first, 1)
        np.add.at(neighbor_counts[:, 0], second, 1)

    average_neighbor_velocity = velocity_sum / np.maximum(neighbor_counts, 1)
    viscosity = (average_neighbor_velocity - velocities) * 2.5
    angle_x, angle_y = radians(rot_x), radians(rot_y)
    sin_x, cos_x = np.sin(angle_x), np.cos(angle_x)
    sin_y, cos_y = np.sin(angle_y), np.cos(angle_y)
    rotation = np.array((
        (cos_y, 0, sin_y),
        (sin_x * sin_y, cos_x, -sin_x * cos_y),
        (-cos_x * sin_y, sin_x, cos_x * cos_y),
    ), dtype=np.float32)
    gravity = rotation.T @ np.array((0, -2.8, 0), dtype=np.float32)
    coriolis = -2 * np.cross(angular_velocity, velocities)
    centrifugal = -np.cross(
        angular_velocity,
        np.cross(angular_velocity, positions),
    )

    velocities += (gravity + viscosity + pressure + coriolis + centrifugal) * delta_time
    velocities *= exp(-0.45 * delta_time)
    positions += velocities * delta_time

    for axis in range(3):
        below_wall = positions[:, axis] < -FLUID_WALL_LIMIT
        above_wall = positions[:, axis] > FLUID_WALL_LIMIT
        positions[below_wall, axis] = -FLUID_WALL_LIMIT
        positions[above_wall, axis] = FLUID_WALL_LIMIT
        velocities[below_wall, axis] = np.abs(velocities[below_wall, axis]) * 0.55
        velocities[above_wall, axis] = -np.abs(velocities[above_wall, axis]) * 0.55

def draw_fluid_particles(positions):
    glPointSize(5)
    glColor3f(0.12, 0.72, 1.0)
    glBegin(GL_POINTS)
    for position in positions:
        glVertex3fv(position)
    glEnd()

def smooth_angle(current, target, factor):
    delta = (target - current + 180) % 360 - 180
    return current + delta * factor

def hand_openness(hand_landmarks, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    palm_center = landmarks[9]

    def distance(first, second):
        dx = (first.x - second.x) * frame_width
        dy = (first.y - second.y) * frame_height
        dz = (first.z - second.z) * frame_width
        return sqrt(dx * dx + dy * dy + dz * dz)

    palm_length = distance(wrist, palm_center)
    if palm_length == 0:
        return 0.0

    finger_extensions = []
    for tip_index in (8, 12, 16, 20):
        extension_ratio = distance(wrist, landmarks[tip_index]) / palm_length
        extension = (extension_ratio - FIST_EXTENSION_RATIO) / (
            OPEN_EXTENSION_RATIO - FIST_EXTENSION_RATIO
        )
        finger_extensions.append(max(0.0, min(1.0, extension)))

    return sum(finger_extensions) / len(finger_extensions)

def draw_camera_background(texture, display):
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], 0, display[1], -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(display[0], 0)
    glTexCoord2f(1, 0)
    glVertex2f(display[0], display[1])
    glTexCoord2f(0, 0)
    glVertex2f(0, display[1])
    glEnd()
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

def main():
    # Inicializar Pygame y OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    near_clip, far_clip = 0.1, 50.0
    top = near_clip * tan(radians(45) / 2)
    right = top * display[0] / display[1]
    glFrustum(-right, right, -top, top, near_clip, far_clip)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)

    camera_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, camera_texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    camera_texture_size = None
    fluid_positions, fluid_velocities = create_fluid_particles()

    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    cap = cv2.VideoCapture(0)

    rot_x, rot_y = 0, 0
    target_rot_x, target_rot_y = rot_x, rot_y
    previous_rot_x, previous_rot_y = rot_x, rot_y
    camera_distance = 5.0
    target_camera_distance = camera_distance
    clock = pygame.time.Clock()

    running = True
    while running:
        delta_time = clock.tick(60) / 1000
        # Eventos Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (K_q, K_ESCAPE):
                running = False

        ret, frame = cap.read()
        if not ret:
            break

        # Espejear frame y convertir a RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar esqueleto de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener la posición de la punta del dedo índice (Landmark 8)
                index_finger = hand_landmarks.landmark[8]

                # Mapeo a ángulos de rotación
                target_rot_x = (0.5 - index_finger.y) * 360
                target_rot_y = (index_finger.x - 0.5) * 360
                openness = hand_openness(hand_landmarks, frame.shape)
                target_camera_distance = CLOSED_FIST_DISTANCE + openness * (
                    OPEN_HAND_DISTANCE - CLOSED_FIST_DISTANCE
                )

        smoothing = 1 - exp(-12 * delta_time)
        rot_x = smooth_angle(rot_x, target_rot_x, smoothing)
        rot_y = smooth_angle(rot_y, target_rot_y, smoothing)
        camera_distance += (target_camera_distance - camera_distance) * smoothing
        if delta_time > 0:
            angular_velocity = np.array((
                radians((rot_x - previous_rot_x) / delta_time),
                radians((rot_y - previous_rot_y) / delta_time),
                0,
            ), dtype=np.float32)
            angular_speed = np.linalg.norm(angular_velocity)
            if angular_speed > 4:
                angular_velocity *= 4 / angular_speed
        else:
            angular_velocity = np.zeros(3, dtype=np.float32)
        previous_rot_x, previous_rot_y = rot_x, rot_y
        step_fluid(
            fluid_positions, fluid_velocities, delta_time,
            rot_x, rot_y, angular_velocity,
        )

        camera_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = camera_frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, camera_texture)
        if camera_texture_size != (frame_width, frame_height):
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, frame_width, frame_height,
                0, GL_RGB, GL_UNSIGNED_BYTE, camera_frame
            )
            camera_texture_size = (frame_width, frame_height)
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, frame_width, frame_height,
                GL_RGB, GL_UNSIGNED_BYTE, camera_frame
            )

        # Renderizar escena 3D en OpenGL
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_camera_background(camera_texture, display)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -camera_distance)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        draw_fluid_particles(fluid_positions)
        glColor3f(1.0, 1.0, 1.0)
        Cube()
        glPopMatrix()

        pygame.display.flip()

    # Liberar recursos
    cap.release()
    hands.close()
    glDeleteTextures([camera_texture])
    pygame.quit()

if __name__ == "__main__":
    main()