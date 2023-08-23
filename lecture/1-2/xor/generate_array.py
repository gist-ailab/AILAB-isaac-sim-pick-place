import pygame 

def in_bound(column_index, row_index, COLUMN_COUNT, ROW_COUNT):
    if (0 <= column_index < COLUMN_COUNT and 0 <= row_index < ROW_COUNT):
        return True
    else:
        return False

def runGame(clock, screen, grid, COLUMN_COUNT, ROW_COUNT, CELL_SIZE):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    
    done = True
    while done: 
        clock.tick(30) 
        screen.fill(BLACK) 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                column_index = event.pos[0] // CELL_SIZE
                row_index = event.pos[1] // CELL_SIZE
                if event.button == 1:
                    if in_bound(column_index, row_index, COLUMN_COUNT, ROW_COUNT):
                        grid[row_index][column_index] = 1
                elif event.button == 3:
                    if in_bound(column_index, row_index, COLUMN_COUNT, ROW_COUNT):
                        grid[row_index][column_index]= 2


        for column_index in range(COLUMN_COUNT):
            for row_index in range(ROW_COUNT):
                tile = grid[row_index][column_index]
                
                if tile == 1:
                    pygame.draw.rect(screen, RED, pygame.Rect(column_index * CELL_SIZE, row_index * CELL_SIZE, CELL_SIZE, CELL_SIZE)) #커버
                elif tile == 2: 
                    pygame.draw.rect(screen, YELLOW, pygame.Rect(column_index * CELL_SIZE, row_index * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    
                pygame.draw.rect(screen, WHITE, pygame.Rect(column_index * CELL_SIZE, row_index * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
        pygame.display.update() 

def get_grid():
    
    pygame.init() 

    
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1000
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    CELL_SIZE = 50
    COLUMN_COUNT = SCREEN_WIDTH // CELL_SIZE
    ROW_COUNT = SCREEN_HEIGHT // CELL_SIZE

    grid = [[0 for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT)]
    
    

    clock = pygame.time.Clock() 
    runGame(clock, screen, grid, COLUMN_COUNT, ROW_COUNT, CELL_SIZE)
    
    pygame.quit() 
    grid = grid[::-1]
    
    return grid