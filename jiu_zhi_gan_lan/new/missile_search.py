from core import *

def validity_time(missile, target_pos, cloud, t_blast):
    the_validity_time = 0.0
    time_step = 0.1
    start_time = 0
    end_time = 20
    current_time = start_time
    m_pre_pos = missile(t_blast)
    c_pre_pos = cloud(t_blast)

    while current_time < end_time:
        t_global = t_blast + current_time  # 全局时间往前跑
        m_pos = missile(t_global)
        c_pos = cloud(t_global)

        # 计算点到直线单位向量
        m_to_t = (target_pos - m_pos) / np.linalg.norm(target_pos - m_pos)

        l = point_to_line_distance(c_pos, m_pos, m_to_t)
        is_between = is_cloud_between(m_pos, c_pos, target_pos)
        is_through = missile_intersect_smoke(m_pre_pos, m_pos, c_pre_pos, c_pos)
        in_c = np.linalg.norm(m_pos - c_pos) <= 10
        if (l <= 10 and is_between) or is_through or in_c:
            the_validity_time += time_step
        m_pre_pos = m_pos
        c_pre_pos = c_pos
        current_time += time_step
    return the_validity_time
def validity_time_set(missile, target_pos, cloud, t_blast):
    the_validity_time = set()
    time_step = 0.1
    start_time = 0
    end_time = 20
    current_time = start_time
    m_pre_pos = missile(t_blast)
    c_pre_pos = cloud(t_blast)

    while current_time < end_time:
        t_global = t_blast + current_time  # 全局时间往前跑
        m_pos = missile(t_global)
        c_pos = cloud(t_global)

        # 计算点到直线单位向量
        m_to_t = (target_pos - m_pos) / np.linalg.norm(target_pos - m_pos)

        l = point_to_line_distance(c_pos, m_pos, m_to_t)
        is_between = is_cloud_between(m_pos, c_pos, target_pos)
        is_through = missile_intersect_smoke(m_pre_pos, m_pos, c_pre_pos, c_pos)
        in_c = np.linalg.norm(m_pos - c_pos) <= 10
        if (l <= 10 and is_between) or is_through or in_c:
            the_validity_time.add(round(t_global, 1))
        m_pre_pos = m_pos
        c_pre_pos = c_pos
        current_time += time_step
    return the_validity_time
def is_cloud_between(m_pos, c_pos, t_pos):
    m_to_c = c_pos - m_pos
    m_to_t = t_pos - m_pos
    if np.linalg.norm(m_to_c) == 0 or np.linalg.norm(m_to_t) == 0:
        return None
    dot_cos_angle = np.dot(m_to_c, m_to_t) / (np.linalg.norm(m_to_c) * np.linalg.norm(m_to_t))
    return dot_cos_angle > 0

def point_to_line_distance(p, a, n):
    w = p - a
    return np.linalg.norm(np.cross(w, n)) / np.linalg.norm(n)

def missile_intersect_smoke(
        m_prev_pos,     # 导弹上一时刻位置
        m_pos,          # 导弹当前位置
        c_prev_pos,     # 烟幕上一时刻中心
        c_pos,          # 烟幕当前中心
):
    """
        判断导弹在当前时间步内是否穿过烟幕（球形）区域。
        采用“相对运动+最近点投影”算法，兼顾效率与精度。
        """
    radius = 10
    missile_move = m_pos - m_prev_pos
    missile_move_length = np.linalg.norm(missile_move)
    smoke_move = c_pos - c_prev_pos

    relative_move = missile_move - smoke_move
    relative_move_length = np.linalg.norm(relative_move)

    if missile_move_length == 0:
        return False
    if relative_move_length == 0:
        return np.linalg.norm(m_pos - c_pos) <= radius

    relative_direction = relative_move / relative_move_length
    c_to_m = m_prev_pos - c_prev_pos
    projection = np.dot(c_to_m, relative_direction)
    perpendicular_vec = c_to_m - projection * relative_direction
    perpendicular_dist = np.linalg.norm(perpendicular_vec)

    if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
        return True
    if np.linalg.norm(m_prev_pos - c_prev_pos) <= radius:
        return True
    if np.linalg.norm(m_pos - c_pos) <= radius:
        return True

    return False