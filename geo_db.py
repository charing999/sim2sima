# geo_db.py (Hybrid Terrain Analyzer)
"""
?섏뒪??湲곕컲 吏??遺꾩꽍 - ?깅뒫 理쒖쟻??踰꾩쟾
- ?섏뒪???뚯씪 ?쒕쾲 ?닿퀬 ?ъ궗??(?대옒??湲곕컲)
- 醫뚰몴 蹂?섍린 罹먯떛
- ?먮윭 ?꾩쟻 泥섎━
- LULC mode + top-k 遺꾪룷

?ъ슜: ???쒖옉 ??TerrainAnalyzer ?몄뒪?댁뒪 ?앹꽦, 醫낅즺 ??close()
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
# ?곗씠??寃쎈줈 ?ㅼ젙 DEM, Slope, Aspect, LULC
# -----------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.environ.get("DEM_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_dem_3857.tif"))
SLOPE_PATH = os.environ.get("SLOPE_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_slope_3857.tif"))
ASPECT_PATH = os.environ.get("ASPECT_PATH", os.path.join(_BASE, "DSK_2026", "DEM", "base_aspect_3857.tif"))
LULC_PATH = os.environ.get("LULC_PATH", os.path.join(_BASE, "DSK_2026", "LC", "base_raster", "base_l3_code_3857.tif"))
# ?좎??쇰났 肄붾뱶 -> ?쇰꺼 (?섍꼍遺 遺꾨쪟泥닿퀎)
# ?遺꾨쪟 (1~7) + ?몃텇瑜?(3?먮━)
LULC_LABEL = {
    # ?遺꾨쪟
    1: "?쒓??붿???,
    2: "?띿뾽吏??,
    3: "?곕┝吏??,
    4: "珥덉?",
    5: "?듭?",
    6: "?섏?",
    7: "?섏뿭",
    
    # ?쒓??붿???(1xx)
    110: "二쇨굅吏??,
    120: "怨듭뾽吏??,
    130: "?곸뾽吏??,
    140: "臾명솕泥댁쑁?댁뼇?쒖꽕",
    150: "援먰넻?쒖꽕",
    151: "?꾨줈",
    152: "泥좊룄",
    153: "??쭔",
    154: "怨듯빆",
    160: "怨듦났?쒖꽕",
    
    # ?띿뾽吏??(2xx)
    210: "??,
    220: "諛?,
    230: "?쒖꽕?щ같吏",
    240: "怨쇱닔??,
    250: "湲고??щ같吏",
    
    # ?곕┝吏??(3xx)
    310: "?쒖뿽?섎┝",
    311: "?쒖뿽?섎┝",
    320: "移⑥뿽?섎┝",
    321: "移⑥뿽?섎┝",
    330: "?쇳슚由?,
    331: "?쇳슚由?,
    
    # 珥덉? (4xx)
    410: "?먯뿰珥덉?",
    420: "?멸났珥덉?",
    421: "怨⑦봽??,
    422: "臾섏?",
    423: "湲고?珥덉?",
    
    # ?듭? (5xx)
    510: "?대쪠?듭?",
    520: "?곗븞?듭?",
    
    # ?섏? (6xx)
    610: "?먯뿰?섏?",
    620: "?멸났?섏?",
    
    # ?섏뿭 (7xx)
    710: "?대쪠??,
    720: "?댁뼇??,
}


def _nanmean_round(x: np.ndarray, ndigits: int = 1, default: float = 0.0) -> float:
    v = np.nanmean(x)
    return default if np.isnan(v) else round(float(v), ndigits)


def analyze_3x3_grid(data: np.ndarray) -> Dict[str, float]:
    """諛곗뿴??3x3濡??섎늻??媛?援ъ뿭 ?됯퇏 怨꾩궛"""
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
    """?뺤닔 肄붾뱶 諛곗뿴?먯꽌 理쒕퉰媛?諛??곸쐞 遺꾪룷 諛섑솚"""
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
    ?섏뒪???뚯씪???쒕쾲 ?닿퀬 ?ъ궗?⑺븯??吏??遺꾩꽍湲?
    ?ъ슜踰?
        analyzer = TerrainAnalyzer()
        result = analyzer.analyze(lat, lon, radius_m=500)
        analyzer.close()  # ??醫낅즺 ??
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

        # ?섏뒪???뚯씪 ?닿린
        self._open_datasets()

    def _open_datasets(self):
        """?섏뒪???뚯씪 ?닿린 諛?蹂?섍린 珥덇린??""
        errors = []
        
        for name, path in self._paths.items():
            if os.path.exists(path):
                try:
                    self._datasets[name] = rasterio.open(path)
                except Exception as e:
                    errors.append(f"{name}: {e}")
            else:
                errors.append(f"{name}: ?뚯씪 ?놁쓬")

        # CRS 媛?몄삤湲?(DEM 湲곗?)
        if "dem" in self._datasets:
            self._raster_crs = self._datasets["dem"].crs
            self._to_raster = Transformer.from_crs("EPSG:4326", self._raster_crs, always_xy=True)
            self._to_wgs84 = Transformer.from_crs(self._raster_crs, "EPSG:4326", always_xy=True)
        
        if errors:
            print(f"[TerrainAnalyzer] 寃쎄퀬: {', '.join(errors)}")

    def close(self):
        """?섏뒪???뚯씪 ?リ린"""
        for ds in self._datasets.values():
            try:
                ds.close()
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _mask_one(self, src, geom) -> Tuple[np.ndarray, Any, Dict[str, float]]:
        """?섏뒪?곗뿉??geometry ?곸뿭 異붿텧 + ?곗씠???덉쭏 ?뺣낫"""
        img, out_transform = mask(src, geom, crop=True)
        arr = img[0].astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        
        # ?곗씠???덉쭏 怨꾩궛
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
        """?쎌? 醫뚰몴瑜?WGS84濡?蹂??""
        x, y = rasterio.transform.xy(out_transform, row, col)
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def _calc_distance_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, str]:
        """??醫뚰몴 媛?嫄곕━(m)? 諛⑹쐞媛?怨꾩궛"""
        import math
        
        # Haversine formula for distance
        R = 6371000  # 吏援?諛섍꼍 (誘명꽣)
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
        """理쒓퀬/理쒖????뺣낫 異붿텧 (以묒떖??湲곗? 嫄곕━/諛⑹쐞 ?ы븿)"""
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        try:
            idx = np.nanargmax(arr) if which == "max" else np.nanargmin(arr)
            row, col = np.unravel_index(idx, arr.shape)
            val = float(arr[row, col])
            lon, lat = self._pixel_to_wgs84(row, col, out_transform)
            
            # 以묒떖??湲곗? 嫄곕━/諛⑹쐞 怨꾩궛
            dist_m, bearing_deg, bearing_cardinal = self._calc_distance_bearing(center_lat, center_lon, lat, lon)
            
            return {
                "val": round(val, 2),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "distance_m_from_center": dist_m,
                "bearing_deg_from_center": bearing_deg,
                "bearing_cardinal_from_center": bearing_cardinal
            }
        except:
            return None

    def analyze(self, lat: float, lon: float, radius_m: float = 500.0, topk_lulc: int = 3) -> Dict[str, Any]:
        """
        醫뚰몴 湲곕컲 吏??遺꾩꽍 - JSON ?뺣떟吏 ?앹꽦
        
        Args:
            lat: ?꾨룄 (WGS84)
            lon: 寃쎈룄 (WGS84)
            radius_m: 遺꾩꽍 諛섍꼍 (誘명꽣)
            topk_lulc: LULC ?곸쐞 遺꾪룷 媛쒖닔
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

        # 蹂?섍린 ?놁쑝硫??먮윭
        if self._to_raster is None:
            result["error"] = "?섏뒪???뚯씪 ?놁쓬"
            return result

        # 踰꾪띁 ?앹꽦
        try:
            cx, cy = self._to_raster.transform(lon, lat)
            geom = [mapping(Point(cx, cy).buffer(radius_m))]
        except Exception as e:
            result["error"] = f"醫뚰몴 蹂???ㅽ뙣: {e}"
            return result

        # DEM 遺꾩꽍 (Digital Elevation Model)
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

        # Slope 遺꾩꽍
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

        # Aspect 遺꾩꽍
        if "aspect" in self._datasets:
            try:
                aspect_arr, _, aspect_quality = self._mask_one(self._datasets["aspect"], geom)
                result["data_quality"]["aspect"] = aspect_quality
                if aspect_arr.size > 0:
                    result["zonal_3x3_avg"]["aspect"] = analyze_3x3_grid(aspect_arr)
            except Exception as e:
                errors.append(f"Aspect: {e}")

        # LULC 遺꾩꽍 (?좎? ?쇰났 遺꾨쪟 肄붾뱶)
        if "lulc" in self._datasets:
            try:
                lulc_arr, lulc_tr, lulc_quality = self._mask_one(self._datasets["lulc"], geom)
                result["data_quality"]["lulc"] = lulc_quality
                mode_code, top = _mode_and_topk(lulc_arr, topk=topk_lulc)
                result["lulc"]["dominant_code"] = mode_code
                result["lulc"]["dominant_label"] = self._lulc_label_map.get(mode_code) if mode_code else None
                
                # dominant_centroid 怨꾩궛: 媛??留롮? ?좎??쇰났??以묒떖??
                if mode_code is not None:
                    try:
                        # ?대떦 肄붾뱶媛 ?덈뒗 ?쎌??ㅼ쓽 ?꾩튂 李얘린
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
                    except:
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
# ?꾩뿭 ?몄뒪?댁뒪 (???쒖옉 ??珥덇린??
# -----------------------------
_ANALYZER: Optional[TerrainAnalyzer] = None


def init_analyzer():
    """吏??遺꾩꽍湲?珥덇린??(???쒖옉 ???몄텧)"""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = TerrainAnalyzer()
        print("[GEO] TerrainAnalyzer 珥덇린???꾨즺")
    return _ANALYZER


def get_terrain_analysis(lat: float, lon: float, radius_m: float = 500) -> Dict[str, Any]:
    """
    湲곗〈 API ?명솚 ?⑥닔 - ?대??곸쑝濡?TerrainAnalyzer ?ъ슜
    """
    global _ANALYZER
    if _ANALYZER is None:
        init_analyzer()
    return _ANALYZER.analyze(lat, lon, radius_m)


def close_analyzer():
    """吏??遺꾩꽍湲?醫낅즺 (??醫낅즺 ???몄텧)"""
    global _ANALYZER
    if _ANALYZER:
        _ANALYZER.close()
        _ANALYZER = None


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """?⑥씪 吏?먯쓽 DEM 怨좊룄 諛섑솚 (誘명꽣)"""
    global _ANALYZER
    if _ANALYZER is None:
        init_analyzer()
    
    if "dem" not in _ANALYZER._datasets or _ANALYZER._to_raster is None:
        return None
    
    try:
        # WGS84 ???섏뒪??醫뚰몴 蹂??
        x, y = _ANALYZER._to_raster.transform(lon, lat)
        
        # ?섏뒪?곗뿉??媛??쎄린
        dem = _ANALYZER._datasets["dem"]
        row, col = dem.index(x, y)
        
        # 踰붿쐞 泥댄겕
        if 0 <= row < dem.height and 0 <= col < dem.width:
            val = dem.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            if dem.nodata is not None and val == dem.nodata:
                return None
            return float(val)
        return None
    except:
        return None


# ?뚯뒪??
if __name__ == "__main__":
    import json
    print("=== Hybrid Terrain Analyzer ?뚯뒪??===")
    
    init_analyzer()
    
    # ?덉뼇/?몃뜒??醫뚰몴
    lat, lon = 37.40, 126.97
    result = get_terrain_analysis(lat, lon, radius_m=500)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    close_analyzer()

