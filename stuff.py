import pygame
import sys
import random
import os

# 初始化Pygame
pygame.init()

# 常量定义
WIDTH, HEIGHT = 800, 600
FPS = 60
BASE_FISH_SIZE = (50, 30)  # 基础尺寸
FISH_COLOR = (88, 88, 222)  # 鱼的颜色
EYE_COLOR = (255, 255, 255)  # 眼睛白色
PUPIL_COLOR = (0, 0, 0)  # 瞳孔黑色
MOUTH_COLOR = (255, 255, 255)  # 嘴巴颜色
FOOD_SIZE = (20, 20)       # 食物尺寸
FOOD_COLOR = (255, 165, 0)  # 食物颜色（橙色）
BACKGROUND_COLOR = (30, 144, 255)  # 海洋背景色
FISH_BASE_SPEED = 3        # 基础速度
FISH_MAX_SPEED = 8         # 最大速度
FISH_CONTROL_FACTOR = 0.5  # 玩家控制力度（值越小控制越轻微）
GROWTH_FACTOR = 0.1        # 成长因子
SCORE_INCREMENT = 10       # 得分增量
GAME_DURATION = 60         # 游戏时长（秒）
HIGHSCORE_FILE = "highscore.txt"  # 最高分记录文件

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fish")

# 字体设置
font = pygame.font.SysFont(None, 36)
control_font = pygame.font.SysFont(None, 24)
timer_font = pygame.font.SysFont(None, 48)
gameover_font = pygame.font.SysFont(None, 72)

# 鱼类（包含成长和控制逻辑）
class Fish:
    def __init__(self):
        self.size = 1.0  # 尺寸因子
        self.speed = FISH_BASE_SPEED  # 移动速度
        self.rect = pygame.Rect(
            random.randint(0, WIDTH - BASE_FISH_SIZE[0]),
            random.randint(0, HEIGHT - BASE_FISH_SIZE[1]),
            *BASE_FISH_SIZE
        )
        self.auto_speed_x = random.choice([-self.speed, self.speed])
        self.auto_speed_y = random.choice([-self.speed, self.speed])
        self.player_speed_x = 0  # 玩家控制的速度分量
        self.player_speed_y = 0  # 玩家控制的速度分量

    def handle_input(self, keys):
        """处理玩家输入"""
        self.player_speed_x = 0
        self.player_speed_y = 0
        
        if keys[pygame.K_a]:  # 左
            self.player_speed_x = -self.speed * FISH_CONTROL_FACTOR
        if keys[pygame.K_d]:  # 右
            self.player_speed_x = self.speed * FISH_CONTROL_FACTOR
        if keys[pygame.K_w]:  # 上
            self.player_speed_y = -self.speed * FISH_CONTROL_FACTOR
        if keys[pygame.K_s]:  # 下
            self.player_speed_y = self.speed * FISH_CONTROL_FACTOR

    def move(self):
        """处理鱼的移动与边界反弹"""
        # 计算总速度（自动移动 + 玩家控制）
        total_speed_x = self.auto_speed_x + self.player_speed_x
        total_speed_y = self.auto_speed_y + self.player_speed_y
        
        # 应用移动
        self.rect.x += total_speed_x
        self.rect.y += total_speed_y
        
        # 边界反弹逻辑
        if self.rect.left < 0:
            self.rect.left = 0
            self.auto_speed_x = abs(self.auto_speed_x)  # 向右反弹
        elif self.rect.right > WIDTH:
            self.rect.right = WIDTH
            self.auto_speed_x = -abs(self.auto_speed_x)  # 向左反弹
            
        if self.rect.top < 0:
            self.rect.top = 0
            self.auto_speed_y = abs(self.auto_speed_y)  # 向下反弹
        elif self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            self.auto_speed_y = -abs(self.auto_speed_y)  # 向上反弹

    def draw(self, surface):
        """绘制带尺寸缩放的鱼，包含眼睛和嘴巴"""
        # 计算缩放后的尺寸
        scaled_size = (
            BASE_FISH_SIZE[0] * self.size,
            BASE_FISH_SIZE[1] * self.size
        )
        scaled_rect = self.rect.inflate(scaled_size[0]-self.rect.width, scaled_size[1]-self.rect.height)
        
        # 绘制鱼的身体
        pygame.draw.ellipse(surface, FISH_COLOR, scaled_rect)
        
        # 确定鱼的方向（基于移动方向）
        facing_right = self.auto_speed_x + self.player_speed_x >= 0
        
        # 眼睛大小和位置计算
        eye_size = max(3, int(scaled_size[1] * 0.2))  # 眼睛大小为鱼高度的20%
        pupil_size = max(1, int(eye_size * 0.5))      # 瞳孔大小为眼睛的50%
        
        # 嘴巴大小和位置计算
        mouth_height = max(4, int(scaled_size[1] * 0.15))  # 嘴巴高度
        mouth_width = max(6, int(scaled_size[0] * 0.25))   # 嘴巴宽度
        
        # 嘴巴位置在鱼的底部中央
        mouth_y = scaled_rect.bottom - mouth_height
        mouth_center_x = scaled_rect.centerx
        
        # 眼睛位置在嘴巴上方两侧
        eye_y = mouth_y - eye_size - max(2, int(scaled_size[1] * 0.05))  # 眼睛在嘴巴上方
        eye_distance_from_mouth = max(5, int(scaled_size[0] * 0.15))     # 眼睛到嘴巴中心的距离
        
        # 根据鱼的朝向调整眼睛位置
        if facing_right:
            left_eye_x = mouth_center_x - eye_distance_from_mouth
            right_eye_x = mouth_center_x + eye_distance_from_mouth
        else:
            left_eye_x = mouth_center_x + eye_distance_from_mouth
            right_eye_x = mouth_center_x - eye_distance_from_mouth
        
        # 绘制眼睛和瞳孔
        pygame.draw.circle(surface, EYE_COLOR, (left_eye_x, eye_y), eye_size)
        pygame.draw.circle(surface, EYE_COLOR, (right_eye_x, eye_y), eye_size)
        pygame.draw.circle(surface, PUPIL_COLOR, (left_eye_x, eye_y), pupil_size)
        pygame.draw.circle(surface, PUPIL_COLOR, (right_eye_x, eye_y), pupil_size)
        
        # 绘制嘴巴（倒置的三角形，位于鱼的底部中央）
        if facing_right:
            mouth_points = [
                (mouth_center_x - mouth_width // 2, mouth_y),
                (mouth_center_x + mouth_width // 2, mouth_y),
                (mouth_center_x, mouth_y + mouth_height)
            ]
        else:
            mouth_points = [
                (mouth_center_x - mouth_width // 2, mouth_y),
                (mouth_center_x + mouth_width // 2, mouth_y),
                (mouth_center_x, mouth_y + mouth_height)
            ]
        
        pygame.draw.polygon(surface, MOUTH_COLOR, mouth_points)

    def eat(self, food):
        """处理吃鱼逻辑"""
        if self.rect.colliderect(food.rect):
            self.size += GROWTH_FACTOR  # 增加尺寸
            # 速度随体型增长，但有上限
            self.speed = min(FISH_BASE_SPEED * (1 + 0.2 * (self.size - 1)), FISH_MAX_SPEED)
            # 重置自动速度分量
            self.auto_speed_x = random.choice([-self.speed, self.speed])
            self.auto_speed_y = random.choice([-self.speed, self.speed])
            return True  # 成功进食
        return False

# 食物类
class FishFood:
    def __init__(self):
        self.rect = pygame.Rect(
            random.randint(0, WIDTH - FOOD_SIZE[0]),
            random.randint(0, HEIGHT - FOOD_SIZE[1]),
            *FOOD_SIZE
        )

    def respawn(self):
        """重新生成食物位置"""
        self.rect.x = random.randint(0, WIDTH - FOOD_SIZE[0])
        self.rect.y = random.randint(0, HEIGHT - FOOD_SIZE[1])

    def draw(self, surface):
        """绘制食物"""
        pygame.draw.rect(surface, FOOD_COLOR, self.rect)

# 加载最高分
def load_highscore():
    try:
        if os.path.exists(HIGHSCORE_FILE):
            with open(HIGHSCORE_FILE, 'r') as file:
                return int(file.read().strip())
        return 0
    except:
        return 0

# 保存最高分
def save_highscore(score):
    try:
        with open(HIGHSCORE_FILE, 'w') as file:
            file.write(str(score))
    except:
        pass

# 游戏主循环
def main():
    clock = pygame.time.Clock()
    fish = Fish()
    food = FishFood()
    score = 0
    highscore = load_highscore()
    
    # 计时器初始化
    start_time = pygame.time.get_ticks()
    game_running = True

    # 游戏控制提示
    control_hints = [
        "Controls: WASD - Slightly change direction",
        "Goal: Become obese",
        "Timing: 1 Minute"
    ]

    while True:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # 游戏结束后按任意键重新开始
                if not game_running:
                    fish = Fish()
                    food = FishFood()
                    score = 0
                    start_time = pygame.time.get_ticks()
                    game_running = True

        if game_running:
            # 计算剩余时间
            elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
            remaining_time = max(0, GAME_DURATION - elapsed_time)
            
            # 游戏结束检查
            if remaining_time == 0:
                game_running = False
                if score > highscore:
                    highscore = score
                    save_highscore(highscore)

            # 处理键盘输入
            keys = pygame.key.get_pressed()
            fish.handle_input(keys)

            # 鱼的逻辑
            fish.move()
            if fish.eat(food):
                score += SCORE_INCREMENT  # 增加得分
                food.respawn()  # 重新生成食物

        # 绘制画面
        screen.fill(BACKGROUND_COLOR)  # 绘制海洋背景
        
        if game_running:
            # 绘制游戏元素
            fish.draw(screen)
            food.draw(screen)
            
            # 绘制得分
            score_text = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            
            # 绘制最高分
            highscore_text = font.render(f"High Score: {highscore}", True, (255, 255, 255))
            screen.blit(highscore_text, (10, 50))
            
            # 绘制计时器
            timer_text = timer_font.render(f"{remaining_time // 60:02d}:{remaining_time % 60:02d}", 
                                           True, (255, 255, 255))
            screen.blit(timer_text, (WIDTH // 2 - timer_text.get_width() // 2, 10))
            
            # 绘制控制提示
            for i, hint in enumerate(control_hints):
                hint_text = control_font.render(hint, True, (255, 255, 255))
                screen.blit(hint_text, (10, HEIGHT - 30 * (len(control_hints) - i)))
        else:
            # 绘制游戏结束画面
            gameover_text = gameover_font.render("Game Over!", True, (255, 255, 0))
            screen.blit(gameover_text, (WIDTH // 2 - gameover_text.get_width() // 2, 
                                        HEIGHT // 2 - gameover_text.get_height() // 2 - 50))
            
            final_score_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
            screen.blit(final_score_text, (WIDTH // 2 - final_score_text.get_width() // 2, 
                                           HEIGHT // 2))
            
            highscore_text = font.render(f"High Score: {highscore}", True, (255, 255, 255))
            screen.blit(highscore_text, (WIDTH // 2 - highscore_text.get_width() // 2, 
                                         HEIGHT // 2 + 50))
            
            restart_text = control_font.render("Press any button to restart", True, (255, 255, 255))
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, 
                                       HEIGHT // 2 + 100))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()