from f1pred.rules import GP_POINTS, SPRINT_POINTS, points_for, points_table


def test_gp_points_2026():
    assert GP_POINTS == (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
    assert points_for(1) == 25
    assert points_for(10) == 1
    assert points_for(11) == 0


def test_sprint_points_2026():
    assert SPRINT_POINTS == (8, 7, 6, 5, 4, 3, 2, 1)
    assert points_for(1, sprint=True) == 8
    assert points_for(9, sprint=True) == 0


def test_points_table_matches_grid_size():
    table = points_table(22, sprint=False)
    assert len(table) == 22
    assert table[0] == 25 and table[9] == 1 and table[21] == 0
    assert list(points_table(3, sprint=True)) == [8, 7, 6]
