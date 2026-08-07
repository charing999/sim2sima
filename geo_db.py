# geo_db.py (Hybrid Terrain Analyzer)
"""
래스터 기반 지형 분석 - 성능 최적화 버전
- 래스터 파일 한번 열고 재사용 (클래스 기반)
- 좌표 변환기 캐싱
- 에러 누적 처리
- LULC mode + top-k 분포

사용: 앱 시작 시 TerrainAnalyzer 인스턴스 생성, 종료 시 close()
"""

from __future__ import annotations
import os
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from pyproj import Transformer
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# -----------------------------
# 데이터 경로 설정 DEM, Slope, Aspect, LULC
# -----------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.environ.get("DEM_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_dem_3857.tif"))
SLOPE_PATH = os.environ.get("SLOPE_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_slope_3857.tif"))
ASPECT_PATH = os.environ.get("ASPECT_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_aspect_3857.tif"))
LULC_PATH = os.environ.get("LULC_PATH", os.path.join(_BASE, "DSK_2026", "LC", "base_raster", "base_l3_code_3857.tif"))
# 토지피복 코드 -> 라벨 (환경부 분류체계)
# 대분류 (1~7) + 세분류 (3자리)
LULC_LABEL = {
    # 대분류
    1: "시가화지역",
    2: "농업지역",
    3: "산림지역",
    4: "초지",
    5: "습지",
    6: "나지",
    7: "수역",
    
    # 시가화지역 (1xx)
    110: "주거지역",
    120: "공업지역",
    130: "상업지역",
    140: "문화체육휴양시설",
    150: "교통시설",
    151: "도로",
    152: "철도",
    153: "항만",
    154: "공항",
    160: "공공시설",
    
    # 농업지역 (2xx)
    210: "논",
    220: "밭",
    230: "시설재배지",
    240: "과수원",
    250: "기타재배지",
    
    # 산림지역 (3xx)
    310: "활엽수림",
    311: "활엽수림",
    320: "침엽수림",
    321: "침엽수림",
    330: "혼효림",
    331: "혼효림",
    
    # 초지 (4xx)
    410: "자연초지",
    420: "인공초지",
    421: "골프장",
    422: "묘지",
    423: "기타초지",
    
    # 습지 (5xx)
    510: "내륙습지",
    520: "연안습지",
    
    # 나지 (6xx)
    610: "자연나지",
    620: "인공나지",
    
    # 수역 (7xx)
    710: "내륙수",
    720: "해양수",
}


def _nanmean_round(x: np.ndarray, ndigits: int = 1, default: float = 0.0) -> float:
    v = np.nanmean(x)
    return default if np.isnan(v) else round(float(v), ndigits)


def analyze_3x3_grid(data: np.ndarray) -> Dict[str, float]:
    """배열을 3x3로 나누어 각 구역 평균 계산"""
    h, w = data.shape
    if h < 3 or w < 3:
        avg = _nanmean_round(data, 1, 0.0)
        return {"NW": 0.0, "N": 0.0, "NE": 0.0, "W": 0.0, "Center": avg, "E": 0.0, "SW": 0.0, "S": 0.0, "SE": 0.0}

    step_h, step_w = h // 3, w // 3
    labels = ["NW", "N", "NE", "W", "Center", "E", "SW", "S", "SE"]
    out = {}
    idx = 0
    for i in range(3):
        for j in range(3):
            region = data[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
            out[labels[idx]] = _nanmean_round(region, 1, 0.0)
            idx += 1
    return out


def _mode_and_topk(values: np.ndarray, topk: int = 3) -> Tuple[Optional[int], List[Tuple[int, int]]]:
    """정수 코드 배열에서 최빈값 및 상위 분포 반환"""
    if values.size == 0:
        return None, []
    v = values[~np.isnan(values)]
    if v.size == 0:
        return None, []
    v_int = v.astype(np.int64)
    uniq, cnt = np.unique(v_int, return_counts=True)
    if uniq.size == 0:
        return None, []
    order = np.argsort(cnt)[::-1]
    mode_code = int(uniq[order[0]])
    top = [(int(uniq[i]), int(cnt[i])) for i in order[:topk]]
    return mode_code, top


class TerrainAnalyzer:
    """
    래스터 파일을 한번 열고 재사용하는 지형 분석기
    사용법:
        analyzer = TerrainAnalyzer()
        result = analyzer.analyze(lat, lon, radius_m=500)
        analyzer.close()  # 앱 종료 시
    """

    def __init__(
        self,
        dem_path: str = DEM_PATH,
        slope_path: str = SLOPE_PATH,
        aspect_path: str = ASPECT_PATH,
        lulc_path: str = LULC_PATH,
        lulc_label_map: Dict[int, str] = None,
    ):
        self._paths = {"dem": dem_path, "slope": slope_path, "aspect": aspect_path, "lulc": lulc_path}
        self._lulc_label_map = lulc_label_map or LULC_LABEL
        self._datasets = {}
        self._to_raster = None
        self._to_wgs84 = None
        self._raster_crs = None

        # 래스터 파일 열기
        self._open_datasets()

    def _open_datasets(self):
        """래스터 파일 열기 및 변환기 초기화"""
        errors = []
        
        for name, path in self._paths.items():
            if os.path.exists(path):
                try:
                    self._datasets[name] = rasterio.open(path)
                except Exception as e:
                    errors.append(f"{name}: {e}")
            else:
                errors.append(f"{name}: 파일 없음")

        # CRS 가져오기 (DEM 기준)
        if "dem" in self._datasets:
            self._raster_crs = self._datasets["dem"].crs
            self._to_raster = Transformer.from_crs("EPSG:4326", self._raster_crs, always_xy=True)
            self._to_wgs84 = Transformer.from_crs(self._raster_crs, "EPSG:4326", always_xy=True)
        
        if errors:
            print(f"[TerrainAnalyzer] 경고: {', '.join(errors)}")

    def close(self):
        """래스터 파일 닫기"""
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _mask_one(self, src, geom) -> Tuple[np.ndarray, Any, Dict[str, float]]:
        """래스터에서 geometry 영역 추출 + 데이터 품질 정보"""
        img, out_transform = mask(src, geom, crop=True)
        arr = img[0].astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        
        # 데이터 품질 계산
        total_pixels = arr.size
        valid_pixels = np.count_nonzero(~np.isnan(arr))
        nodata_pixels = total_pixels - valid_pixels
        quality = {
            "pixel_count": total_pixels,
            "valid_ratio": round(valid_pixels / total_pixels, 3) if total_pixels > 0 else 0.0,
            "nodata_ratio": round(nodata_pixels / total_pixels, 3) if total_pixels > 0 else 1.0,
        }
        return arr, out_transform, quality

    def _pixel_to_wgs84(self, row: int, col: int, out_transform) -> Tuple[float, float]:
        """픽셀 좌표를 WGS84로 변환"""
        x, y = rasterio.transform.xy(out_transform, row, col)
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def _calc_distance_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, str]:
        """두 좌표 간 거리(m)와 방위각 계산"""
        import math
        
        # Haversine formula for distance
        R = 6371000  # 지구 반경 (미터)
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
        distance_m = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Bearing calculation
        y = math.sin(dlam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        bearing_deg = (math.degrees(math.atan2(y, x)) + 360) % 360
        
        # Cardinal direction
        cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        bearing_cardinal = cardinals[int((bearing_deg + 22.5) / 45) % 8]
        
        return round(distance_m, 1), round(bearing_deg, 1), bearing_cardinal

    def _anchor_info(self, arr: np.ndarray, out_transform, which: str, center_lat: float, center_lon: float) -> Optional[Dict]:
        """최고/최저점 정보 추출 (중심점 기준 거리/방위 포함)"""
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        try:
            idx = np.nanargmax(arr) if which == "max" else np.nanargmin(arr)
            row, col = np.unravel_index(idx, arr.shape)
            val = float(arr[row, col])
            lon, lat = self._pixel_to_wgs84(row, col, out_transform)
            
            # 중심점 기준 거리/방위 계산
            dist_m, bearing_deg, bearing_cardinal = self._calc_distance_bearing(center_lat, center_lon, lat, lon)
            
            return {
                "val": round(val, 2),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "distance_m_from_center": dist_m,
                "bearing_deg_from_center": bearing_deg,
                "bearing_cardinal_from_center": bearing_cardinal
            }
        except Exception:
            return None

    def analyze(self, lat: float, lon: float, radius_m: float = 500.0, topk_lulc: int = 3) -> Dict[str, Any]:
        """
        좌표 기반 지형 분석 - JSON 정답지 생성
        
        Args:
            lat: 위도 (WGS84)
            lon: 경도 (WGS84)
            radius_m: 분석 반경 (미터)
            topk_lulc: LULC 상위 분포 개수
        """
        result = {
            "location": {"lat": round(lat, 6), "lon": round(lon, 6)},
            "radius_m": radius_m,
            "anchors": {},
            "zonal_3x3_avg": {},
            "lulc": {},
            "overall_stats": {},
            "data_quality": {},
            "error": "",
        }
        errors = []

        # 변환기 없으면 에러
        if self._to_raster is None:
            result["error"] = "래스터 파일 없음"
            return result

        # 버퍼 생성
        try:
            cx, cy = self._to_raster.transform(lon, lat)
            geom = [mapping(Point(cx, cy).buffer(radius_m))]
        except Exception as e:
            result["error"] = f"좌표 변환 실패: {e}"
            return result

        # DEM 분석 (Digital Elevation Model)
        if "dem" in self._datasets:
            try:
                dem_arr, dem_tr, dem_quality = self._mask_one(self._datasets["dem"], geom)
                result["data_quality"]["dem"] = dem_quality
                if dem_arr.size > 0:
                    hp = self._anchor_info(dem_arr, dem_tr, "max", lat, lon)
                    lp = self._anchor_info(dem_arr, dem_tr, "min", lat, lon)
                    if hp: result["anchors"]["highest_point"] = hp
                    if lp: result["anchors"]["lowest_point"] = lp
                    result["zonal_3x3_avg"]["altitude"] = analyze_3x3_grid(dem_arr)
                    result["overall_stats"]["avg_altitude"] = _nanmean_round(dem_arr)
                    if not np.all(np.isnan(dem_arr)):
                        result["overall_stats"]["min_altitude"] = round(float(np.nanmin(dem_arr)), 1)
                        result["overall_stats"]["max_altitude"] = round(float(np.nanmax(dem_arr)), 1)
            except Exception as e:
                errors.append(f"DEM: {e}")

        # Slope 분석
        if "slope" in self._datasets:
            try:
                slope_arr, slope_tr, slope_quality = self._mask_one(self._datasets["slope"], geom)
                result["data_quality"]["slope"] = slope_quality
                if slope_arr.size > 0:
                    sp = self._anchor_info(slope_arr, slope_tr, "max", lat, lon)
                    if sp: result["anchors"]["steepest_point"] = sp
                    result["zonal_3x3_avg"]["slope"] = analyze_3x3_grid(slope_arr)
                    result["overall_stats"]["avg_slope"] = _nanmean_round(slope_arr)
                    if not np.all(np.isnan(slope_arr)):
                        result["overall_stats"]["max_slope"] = round(float(np.nanmax(slope_arr)), 1)
            except Exception as e:
                errors.append(f"Slope: {e}")

        # Aspect 분석
        if "aspect" in self._datasets:
            try:
                aspect_arr, _, aspect_quality = self._mask_one(self._datasets["aspect"], geom)
                result["data_quality"]["aspect"] = aspect_quality
                if aspect_arr.size > 0:
                    result["zonal_3x3_avg"]["aspect"] = analyze_3x3_grid(aspect_arr)
            except Exception as e:
                errors.append(f"Aspect: {e}")

        # LULC 분석 (토지 피복 분류 코드)
        if "lulc" in self._datasets:
            try:
                lulc_arr, lulc_tr, lulc_quality = self._mask_one(self._datasets["lulc"], geom)
                result["data_quality"]["lulc"] = lulc_quality
                mode_code, top = _mode_and_topk(lulc_arr, topk=topk_lulc)
                result["lulc"]["dominant_code"] = mode_code
                result["lulc"]["dominant_label"] = self._lulc_label_map.get(mode_code) if mode_code else None
                
                # dominant_centroid 계산: 가장 많은 토지피복의 중심점
                if mode_code is not None:
                    try:
                        # 해당 코드가 있는 픽셀들의 위치 찾기
                        mask_pixels = (lulc_arr == mode_code)
                        if np.any(mask_pixels):
                            rows, cols = np.where(mask_pixels)
                            center_row = int(np.mean(rows))
                            center_col = int(np.mean(cols))
                            clon, clat = self._pixel_to_wgs84(center_row, center_col, lulc_tr)
                            result["lulc"]["dominant_centroid"] = {
                                "lat": round(clat, 6),
                                "lon": round(clon, 6)
                            }
                    except Exception:
                        pass
                
                result["lulc"]["distribution"] = [
                    {"code": code, "label": self._lulc_label_map.get(code), "count": cnt}
                    for code, cnt in top
                ]
            except Exception as e:
                errors.append(f"LULC: {e}")

        result["error"] = " | ".join(errors)
        return result


# -----------------------------
# 전역 인스턴스 (앱 시작 시 초기화)
# -----------------------------
_ANALYZER: Optional[TerrainAnalyzer] = None


def init_analyzer():
    """지형 분석기 초기화 (앱 시작 시 호출)"""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = TerrainAnalyzer()
        print("[GEO] TerrainAnalyzer 초기화 완료")
    return _ANALYZER


def get_terrain_analysis(lat: float, lon: float, radius_m: float = 500) -> Dict[str, Any]:
    """
    기존 API 호환 함수 - 내부적으로 TerrainAnalyzer 사용
    """
    global _ANALYZER
    if _ANALYZER is None:
        init_analyzer()
    return _ANALYZER.analyze(lat, lon, radius_m)


def close_analyzer():
    """지형 분석기 종료 (앱 종료 시 호출)"""
    global _ANALYZER
    if _ANALYZER:
        _ANALYZER.close()
        _ANALYZER = None


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """단일 지점의 DEM 고도 반환 (미터)"""
    global _ANALYZER
    if _ANALYZER is None:
        init_analyzer()
    
    if "dem" not in _ANALYZER._datasets or _ANALYZER._to_raster is None:
        return None
    
    try:
        # WGS84 → 래스터 좌표 변환
        x, y = _ANALYZER._to_raster.transform(lon, lat)
        
        # 래스터에서 값 읽기
        dem = _ANALYZER._datasets["dem"]
        row, col = dem.index(x, y)
        
        # 범위 체크
        if 0 <= row < dem.height and 0 <= col < dem.width:
            val = dem.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            if dem.nodata is not None and val == dem.nodata:
                return None
            return float(val)
        return None
    except Exception:
        return None


# 테스트
if __name__ == "__main__":
    import json
    print("=== Hybrid Terrain Analyzer 테스트 ===")
    
    init_analyzer()
    
    # 안양/인덕원 좌표
    lat, lon = 37.40, 126.97
    result = get_terrain_analysis(lat, lon, radius_m=500)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    close_analyzer()
