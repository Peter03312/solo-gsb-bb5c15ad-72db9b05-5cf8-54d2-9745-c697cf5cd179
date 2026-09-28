"""时间线合并与预算算法单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.timeline import (
    Segment,
    attribute_exposure,
    clip_segments,
    coverage_gaps,
    first_exceed_time,
    integrate,
    integrate_within,
    merge_intervals,
    reconcile_illuminance,
    year_bounds,
)


def _dt(h: float, day: int = 1) -> datetime:
    return datetime(2026, 3, day, tzinfo=timezone.utc) + timedelta(hours=h)


def test_disjoint_intervals_kept_separate():
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(4), _dt(6), 100)])
    assert [(s.start, s.end, s.lux) for s in segs] == [
        (_dt(0), _dt(2), 50),
        (_dt(4), _dt(6), 100),
    ]


def test_overlap_takes_max_lux_not_sum():
    # 两盏灯在 10:00-12:00 重叠：有效照度取 max(50, 200)=200
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(10), _dt(14), 200)])
    assert segs == [
        Segment(_dt(8), _dt(10), 50),
        Segment(_dt(10), _dt(14), 200),
    ]
    assert integrate(segs) == 50 * 2 + 200 * 4


def test_triple_overlap_and_coalescing():
    segs = merge_intervals(
        [(_dt(0), _dt(3), 50), (_dt(2), _dt(5), 80), (_dt(4), _dt(6), 80)]
    )
    # 0-2:50, 2-3:max(50,80)=80, 3-4:80, 4-5:max(80,80)=80, 5-6:80
    # 相邻同照度合并：2-6 连成一段 80
    assert segs == [Segment(_dt(0), _dt(2), 50), Segment(_dt(2), _dt(6), 80)]


def test_touching_intervals_are_not_overlap():
    # 首尾相接不算重叠，但同照度会被合并为一个片段
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(4), 50)])
    assert segs == [Segment(_dt(0), _dt(4), 50)]


def test_zero_effective_lux_dropped_when_brighter_lamp_active_then_ends():
    # 0 lux 区间与有灯区间重叠时不产生影响
    segs = merge_intervals([(_dt(0), _dt(4), 0), (_dt(1), _dt(3), 50)])
    assert segs == [Segment(_dt(1), _dt(3), 50)]


def test_clip_to_year_window():
    start, end = year_bounds(2026)
    segs = merge_intervals(
        [
            (datetime(2025, 12, 31, 20, tzinfo=timezone.utc),
             datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
        ]
    )
    clipped = clip_segments(segs, start, end)
    assert clipped == [
        Segment(start, datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
    ]
    assert integrate_within(segs, start, end) == 400


def test_first_exceed_time_interpolated():
    # 恒定 100 lux，上限 250 lux·h → 2.5 小时处超标
    segs = merge_intervals([(_dt(0), _dt(10), 100)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.5)


def test_exactly_at_limit_is_not_exceeded():
    segs = merge_intervals([(_dt(0), _dt(5), 50)])  # 恰好 250
    assert first_exceed_time(segs, 250.0, year_bounds(2026)[0]) is None


def test_first_exceed_skips_low_exposure_segment():
    # 第一段 2h*50=100，第二段从 2h 起 200lux，还需 150 → 0.75h
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(6), 200)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.75)


def test_coverage_gaps_merges_dark_and_dim_periods():
    # 08-12 @50 达标；12-14 无灯、14-16 @30 不达标、16-18 无灯 → 连成单一缺口
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(14), _dt(16), 30)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(12), _dt(18))]
    assert compliant == 4.0


def test_coverage_gaps_exact_threshold_is_compliant():
    # 照度恰好等于门槛 → 达标，无缺口
    segs = merge_intervals([(_dt(8), _dt(18), 50)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == []
    assert compliant == 10.0


def test_coverage_gaps_no_light_at_all():
    # 完全无灯：整个窗口是一个缺口
    gaps, compliant = coverage_gaps([], _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(18))]
    assert compliant == 0.0


def test_coverage_gaps_split_around_bright_period():
    # 10-12 @100 达标，前后各一段缺口
    segs = merge_intervals([(_dt(10), _dt(12), 100)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(10)), (_dt(12), _dt(18))]
    assert compliant == 2.0


def test_coverage_gaps_clips_cross_day_segment_to_window():
    # 跨日片段 20:00 → 次日 10:00：当日窗口 08:00-12:00 无交集，次日窗口取 08:00-10:00
    segs = merge_intervals([(_dt(20), _dt(10, day=2), 80)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(12), 50)
    assert gaps == [(_dt(8), _dt(12))]
    assert compliant == 0.0
    gaps, compliant = coverage_gaps(segs, _dt(8, day=2), _dt(12, day=2), 50)
    assert gaps == [(_dt(10, day=2), _dt(12, day=2))]
    assert compliant == 2.0


# ------------------------------------------------------------- 曝光归因

def test_attribute_exposure_overlap_takes_max_and_shadows():
    # A: 08-18 @50；B: 10-12 @200 → 重叠段取 max，A 被遮蔽 2 小时
    total, attrs = attribute_exposure(
        [(1, _dt(8), _dt(18), 50), (2, _dt(10), _dt(12), 200)], _dt(8), _dt(18)
    )
    assert total == 50 * 2 + 200 * 2 + 50 * 6
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 50 * 2 + 50 * 6
    assert by_id[1].shadowed_hours == 2.0
    assert by_id[2].lux_hours == 200 * 2
    assert by_id[2].shadowed_hours == 0.0


def test_attribute_exposure_tied_brightest_split_evenly():
    # 两条 100 lux 并列最亮：该片的 200 lux·h 平分；50 lux 不获归因只记遮蔽
    total, attrs = attribute_exposure(
        [
            (1, _dt(10), _dt(12), 100),
            (2, _dt(10), _dt(12), 100),
            (3, _dt(10), _dt(12), 50),
        ],
        _dt(8),
        _dt(18),
    )
    assert total == 200.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 100.0
    assert by_id[2].lux_hours == 100.0
    assert by_id[3].lux_hours == 0.0
    assert by_id[3].shadowed_hours == 2.0


def test_attribute_exposure_clips_cross_boundary_intervals():
    # 跨界区间只计入窗口内部分；窗口外部分不产生归因与遮蔽
    total, attrs = attribute_exposure(
        [(1, _dt(20), _dt(10, day=2), 80), (2, _dt(21), _dt(23), 30)],
        _dt(22),
        _dt(23),
    )
    assert total == 80.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 80.0
    assert by_id[1].shadowed_hours == 0.0
    assert by_id[2].lux_hours == 0.0
    assert by_id[2].shadowed_hours == 1.0


def test_attribute_exposure_zero_lux_and_empty_window():
    # 0 lux 区间相交但不获归因也不被遮蔽；窗口内无区间时总量为 0、列表为空
    total, attrs = attribute_exposure([(1, _dt(8), _dt(10), 0)], _dt(8), _dt(18))
    assert total == 0.0
    assert [a.interval_id for a in attrs] == [1]
    assert attrs[0].lux_hours == 0.0
    assert attrs[0].shadowed_hours == 0.0

    total, attrs = attribute_exposure([], _dt(8), _dt(18))
    assert total == 0.0
    assert attrs == []


# ------------------------------------------------------------- 现场照度对账

def test_reconcile_slices_at_all_change_points_and_integrates():
    # 计划：08-16 @50，12-13 @300（重叠取 max）
    planned = merge_intervals(
        [(_dt(8), _dt(16), 50), (_dt(12), _dt(13), 300)]
    )
    # 实测：08-11 @100；11-14 @0；14-18 @350
    readings = [(_dt(8), 100), (_dt(11), 0), (_dt(14), 350)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(18), 50)

    # 正偏差：08-11 +50×3 + 14-16 +300×2 + 16-18 +350×2 = 1450
    assert res.positive_lux_hours == 1450.0
    # 负偏差：11-12 −50 + 12-13 −300 + 13-14 −50 = 400
    assert res.negative_lux_hours == 400.0
    # 超阈（严格 > 50）：12-13（|−300|）+ 14-18（+300/+350）= 5h；
    # 08-11 与 11-12、13-14 恰为 ±50 不算异常
    assert res.anomaly_hours == 5.0
    assert [(p.direction, p.start, p.end) for p in res.periods] == [
        ("under", _dt(12), _dt(13)),
        ("over", _dt(14), _dt(18)),
    ]


def test_reconcile_merges_same_direction_runs_across_change_points():
    # 计划：08-18 全程 100；实测在计划变化点两侧同为 +60 → 合并为一个 over 时段
    planned = merge_intervals(
        [(_dt(8), _dt(13), 100), (_dt(13), _dt(18), 100)]
    )
    readings = [(_dt(8), 160), (_dt(13), 160)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(18), 50)
    assert res.positive_lux_hours == 600.0
    assert res.negative_lux_hours == 0.0
    assert res.anomaly_hours == 10.0
    assert [(p.direction, p.start, p.end) for p in res.periods] == [
        ("over", _dt(8), _dt(18))
    ]


def test_reconcile_opposite_directions_not_merged():
    # 计划全程 100；实测 08-12 @160（over），12-18 @40（under），中间无正常段
    planned = merge_intervals([(_dt(8), _dt(18), 100)])
    readings = [(_dt(8), 160), (_dt(12), 40)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(18), 50)
    assert res.anomaly_hours == 10.0
    assert [(p.direction, p.start, p.end) for p in res.periods] == [
        ("over", _dt(8), _dt(12)),
        ("under", _dt(12), _dt(18)),
    ]


def test_reconcile_normal_gap_splits_same_direction_periods():
    # over → 正常 → over：同方向但不相邻，必须是两个时段
    planned = merge_intervals([(_dt(8), _dt(18), 100)])
    readings = [(_dt(8), 160), (_dt(10), 100), (_dt(12), 160), (_dt(14), 100)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(18), 50)
    assert res.anomaly_hours == 4.0
    assert [(p.start, p.end) for p in res.periods] == [
        (_dt(8), _dt(10)),
        (_dt(12), _dt(14)),
    ]
    assert all(p.direction == "over" for p in res.periods)


def test_reconcile_unplanned_time_counts_as_zero_planned():
    # 计划只有 08-10 @50；窗口 08-14 内其余时段计划照度为 0
    planned = merge_intervals([(_dt(8), _dt(10), 50)])
    readings = [(_dt(8), 50), (_dt(10), 30)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(14), 10)
    # 08-10 实测=计划；10-14 计划 0、实测 30 → over 30×4=120，异常 4h
    assert res.positive_lux_hours == 120.0
    assert res.negative_lux_hours == 0.0
    assert res.anomaly_hours == 4.0
    assert [(p.direction, p.start, p.end) for p in res.periods] == [
        ("over", _dt(10), _dt(14))
    ]

    # 完全无计划照明 + 实测 0：正负偏差均为 0、无异常
    res = reconcile_illuminance([(_dt(8), 0)], [], _dt(8), _dt(18), 10)
    assert res.positive_lux_hours == res.negative_lux_hours == 0.0
    assert res.anomaly_hours == 0.0 and res.periods == ()


def test_reconcile_exactly_at_tolerance_is_not_anomaly():
    # 实测恰好 = 计划 ± 阈值：积分照常，异常时长为 0、无异常时段
    planned = merge_intervals([(_dt(8), _dt(18), 100)])
    readings = [(_dt(8), 150), (_dt(13), 50)]
    res = reconcile_illuminance(readings, planned, _dt(8), _dt(18), 50)
    assert res.positive_lux_hours == 250.0
    assert res.negative_lux_hours == 250.0
    assert res.anomaly_hours == 0.0
    assert res.periods == ()
