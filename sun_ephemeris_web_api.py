import os
import datetime
import uvicorn
import numpy as np
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Astropy and SunPy imports
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import sunpy.coordinates.sun as sun

app = FastAPI(
    title="Nap Efemerisz Web API",
    description="Nap P, B0, L0 efemeriszek és kelés/delelés/nyugvás számítások",
    version="1.0.0"
)

# Fully enabling CORS to allow calls from any local or remote HTML page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_sun_alt_single(t: Time, lat: float, lon: float) -> float:
    """Calculates the altitude of the Sun in degrees for a single point in time."""
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
    sun_coord = get_sun(t)
    altaz_frame = AltAz(obstime=t, location=loc)
    return sun_coord.transform_to(altaz_frame).alt.deg

def get_sun_alt_vector(t_seq: Time, lat: float, lon: float) -> np.ndarray:
    """Calculates the altitude of the Sun in degrees for an array of times (vectorized, very fast)."""
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
    sun_coord = get_sun(t_seq)
    altaz_frame = AltAz(obstime=t_seq, location=loc)
    return sun_coord.transform_to(altaz_frame).alt.deg

def find_crossing(t1: Time, t2: Time, lat: float, lon: float, target_alt: float = -0.8333, tol: float = 1e-6) -> Time:
    """Refines the crossing point where the Sun reaches target_alt using the bisection method."""
    jd1 = t1.jd
    jd2 = t2.jd
    
    val1 = get_sun_alt_single(t1, lat, lon) - target_alt
    val2 = get_sun_alt_single(t2, lat, lon) - target_alt
    
    # Run bisection for 18 steps (precision < 0.01 seconds)
    for _ in range(18):
        jd_mid = 0.5 * (jd1 + jd2)
        t_mid = Time(jd_mid, format='jd', scale='utc')
        val_mid = get_sun_alt_single(t_mid, lat, lon) - target_alt
        
        if abs(val_mid) < tol:
            return t_mid
            
        if val1 * val_mid < 0:
            jd2 = jd_mid
            val2 = val_mid
        else:
            jd1 = jd_mid
            val1 = val_mid
            
    return Time(0.5 * (jd1 + jd2), format='jd', scale='utc')

def find_transit(t1: Time, t2: Time, lat: float, lon: float) -> Time:
    """Finds the precise solar noon (transit) when altitude is maximized using Golden Section Search."""
    gr = (5**0.5 - 1) / 2
    jd_a = t1.jd
    jd_b = t2.jd
    
    jd_c = jd_b - gr * (jd_b - jd_a)
    jd_d = jd_a + gr * (jd_b - jd_a)
    
    for _ in range(16):
        t_c = Time(jd_c, format='jd', scale='utc')
        t_d = Time(jd_d, format='jd', scale='utc')
        
        alt_c = get_sun_alt_single(t_c, lat, lon)
        alt_d = get_sun_alt_single(t_d, lat, lon)
        
        if alt_c > alt_d:
            jd_b = jd_d
            jd_d = jd_c
            jd_c = jd_b - gr * (jd_b - jd_a)
        else:
            jd_a = jd_c
            jd_c = jd_d
            jd_d = jd_a + gr * (jd_b - jd_a)
            
    return Time(0.5 * (jd_a + jd_b), format='jd', scale='utc')

@app.get("/api/ephemeris")
async def calculate_ephemeris(
    date: Optional[str] = Query(None, description="Dátum YYYY-MM-DD formátumban (alapértelmezett: ma)"),
    time: Optional[str] = Query(None, description="Időpont HH:MM:SS formátumban UT szerint (alapértelmezett: 12:00:00)"),
    lat: float = Query(47.4983, description="Észlelő földrajzi szélessége fokban (alapértelmezett: 47.4983 - Budapest)"),
    lon: float = Query(19.0408, description="Észlelő földrajzi hosszúsága fokban (alapértelmezett: 19.0408 - Budapest)")
):
    try:
        # Resolve date
        if not date:
            date_str = datetime.date.today().isoformat()
        else:
            # Validate format
            datetime.date.fromisoformat(date)
            date_str = date
            
        # Resolve time
        if not time:
            time_str = "12:00:00"
        else:
            # Simple format verification
            parts = time.split(':')
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError()
            time_str = time

        # Compute SunPy ephemerides (P, B0, L0) for the target calculation datetime
        target_time_str = f"{date_str} {time_str}"
        t_calc = Time(target_time_str, scale='utc')
        
        p_angle = sun.P(t_calc).to('deg').value
        b0_angle = sun.B0(t_calc).to('deg').value
        l0_angle = sun.L0(t_calc).to('deg').value
        c_rot_raw = sun.carrington_rotation_number(t_calc)
        c_rot_full = getattr(c_rot_raw, "value", c_rot_raw)

        # We scan from 00:00:00 to 24:00:00 UTC using a 48-point grid (every 30 mins)
        base_time = Time(f"{date_str} 00:00:00", scale='utc')
        time_steps = [base_time + (i / 48.0) * u.day for i in range(49)]
        
        # Convert steps to vectorized Astropy Time array
        jds = [t.jd for t in time_steps]
        t_vector = Time(jds, format='jd', scale='utc')
        
        # Fast vectorized altitude calculation
        altitudes = get_sun_alt_vector(t_vector, lat, lon)
        
        # Standard astronomical horizon is -0.8333 degrees (refraction + solar radius correction)
        h_target = -0.8333
        
        sunrise_t = None
        sunset_t = None
        
        # Find crossings
        for i in range(len(altitudes) - 1):
            alt1 = altitudes[i]
            alt2 = altitudes[i+1]
            t1 = time_steps[i]
            t2 = time_steps[i+1]
            
            # Sunrise: crossing from below to above target horizon
            if alt1 <= h_target < alt2:
                sunrise_t = find_crossing(t1, t2, lat, lon, target_alt=h_target)
            
            # Sunset: crossing from above to below target horizon
            if alt1 >= h_target > alt2:
                sunset_t = find_crossing(t1, t2, lat, lon, target_alt=h_target)

        # Transit search: locate maximum altitude step
        idx_max = int(np.argmax(altitudes))
        t_start_transit = time_steps[max(0, idx_max - 1)]
        t_end_transit = time_steps[min(len(time_steps) - 1, idx_max + 1)]
        transit_t = find_transit(t_start_transit, t_end_transit, lat, lon)

        # Handle Polar Day / Night cases
        all_above = all(alt > h_target for alt in altitudes)
        all_below = all(alt < h_target for alt in altitudes)
        
        if all_above:
            sunrise_str = "Nincs kelés (Sarki nappal)"
            sunset_str = "Nincs nyugvás (Sarki nappal)"
        elif all_below:
            sunrise_str = "Nincs kelés (Sarki éjszaka)"
            sunset_str = "Nincs nyugvás (Sarki éjszaka)"
        else:
            sunrise_str = sunrise_t.utc.datetime.strftime("%H:%M:%S UT") if sunrise_t else "Nem meghatározható"
            sunset_str = sunset_t.utc.datetime.strftime("%H:%M:%S UT") if sunset_t else "Nem meghatározható"

        transit_str = transit_t.utc.datetime.strftime("%H:%M:%S UT") if transit_t else "Nem meghatározható"

        response_data = {
            "input": {
                "datum": date_str,
                "idopont_ut": time_str,
                "lat": lat,
                "lon": lon
            },
            "efemeriszek": {
                "P": round(p_angle, 5),
                "B0": round(b0_angle, 5),
                "L0": round(l0_angle, 5),
                "CR": int(c_rot_full),
                "P_formazott": f"{round(p_angle, 4)}°",
                "B0_formazott": f"{'+' if b0_angle >= 0 else ''}{round(b0_angle, 4)}°",
                "L0_formazott": f"{round(l0_angle, 4)}°",
                "CR_formazott": f"{int(c_rot_full)} ({c_rot_full:.4f})",
                "magyarazat": {
                    "P": "Helyzeti szög (a forgástengely északi pólusának dőlése a geocentrikus északtól keletre)",
                    "B0": "Heliofizikai szélesség (a látható korong középpontjának szélessége a napiegyenlítőtől mérve)",
                    "L0": "Heliofizikai hosszúság (a látható korong középpontjának Carrington-hosszúsága)",
                    "CR": "Carrington rotáció"
                }
            },
            "nap_esemenyek_ut": {
                "napkelte": sunrise_str,
                "deletes_del": transit_str,
                "napnyugta": sunset_str,
                "iso": {
                    "napkelte": sunrise_t.utc.isot + "Z" if sunrise_t and not (all_above or all_below) else None,
                    "deletes_del": transit_t.utc.isot + "Z" if transit_t else None,
                    "napnyugta": sunset_t.utc.isot + "Z" if sunset_t and not (all_above or all_below) else None
                }
            }
        }
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hiba a számítás során: {str(e)}. Ellenőrizze a formátumokat (YYYY-MM-DD és HH:MM:SS)!")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nap Efemerisz és Esemény Kalkulátor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {
                background: radial-gradient(circle at center, #111827 0%, #030712 100%);
            }
        </style>
    </head>
    <body class="text-gray-100 min-h-screen py-8 px-4 font-sans selection:bg-orange-500 selection:text-white">
        <div class="max-w-6xl mx-auto">
            <!-- Header -->
            <header class="text-center mb-10">
                <div class="inline-flex items-center justify-center p-3 bg-gradient-to-br from-amber-400 to-orange-600 rounded-full shadow-lg shadow-orange-500/20 mb-4 animate-pulse">
                    <i class="fa-solid fa-sun text-4xl text-white"></i>
                </div>
                <h1 class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-amber-300 via-orange-400 to-rose-500">
                    Nap Efemerisz & Esemény Kalkulátor
                </h1>
                <p class="mt-2 text-gray-400 max-w-lg mx-auto text-sm sm:text-base">
                    Valós idejű heliografikus koordináta-számítás és napkelte/napnyugta meghatározás SunPy & Astropy alapokon.
                </p>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <!-- Inputs Section (Left column) -->
                <div class="lg:col-span-4 bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl backdrop-blur-md">
                    <h2 class="text-xl font-bold mb-4 flex items-center gap-2 text-amber-400 border-b border-gray-800 pb-2">
                        <i class="fa-solid fa-sliders"></i> Paraméterek
                    </h2>
                    
                    <form id="calcForm" class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-1">Dátum</label>
                            <input type="date" id="inputDate" name="date" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                            <p class="text-xs text-gray-500 mt-1">Alapértelmezett: aktuális mai dátum</p>
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-1">Időpont (UT / Világidő)</label>
                            <input type="time" step="1" id="inputTime" name="time" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                            <p class="text-xs text-gray-500 mt-1">Alapértelmezett: déli 12:00:00 UT</p>
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-1">Földrajzi szélesség (Latitude)</label>
                            <input type="number" step="0.0001" id="inputLat" name="lat" value="47.4983" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                            <p class="text-xs text-gray-500 mt-1">Budapest: ~47.4983 fok (Észak pozitív)</p>
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-1">Földrajzi hosszúság (Longitude)</label>
                            <input type="number" step="0.0001" id="inputLon" name="lon" value="19.0408" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                            <p class="text-xs text-gray-500 mt-1">Budapest: ~19.0408 fok (Kelet pozitív)</p>
                        </div>

                        <button type="submit" class="w-full bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white font-bold py-3 px-4 rounded-xl shadow-lg transition duration-200 transform active:scale-95 flex items-center justify-center gap-2">
                            <i class="fa-solid fa-calculator"></i> Számítás Futtatása
                        </button>
                    </form>
                </div>

                <!-- Outputs Section (Right column) -->
                <div class="lg:col-span-8 space-y-6">
                    <!-- Status Alerts -->
                    <div id="loading" class="hidden bg-gray-900/60 border border-amber-500/30 text-amber-200 p-4 rounded-xl flex items-center justify-center gap-3">
                        <i class="fa-solid fa-spinner animate-spin text-xl text-amber-500"></i>
                        <span>Számítás folyamatban a szerveren...</span>
                    </div>

                    <div id="errorBox" class="hidden bg-rose-950/40 border border-rose-500/30 text-rose-200 p-4 rounded-xl flex items-center gap-3">
                        <i class="fa-solid fa-triangle-exclamation text-xl text-rose-500"></i>
                        <span id="errorMessage">Hiba történt.</span>
                    </div>

                    <!-- Main Results Display -->
                    <div id="results" class="space-y-6">
                        <!-- Upper: Visualiser & Ephemeris Cards -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <!-- Canvas Graphics Card -->
                            <div class="bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl flex flex-col items-center justify-center">
                                <h3 class="text-sm font-semibold tracking-wide text-gray-400 uppercase mb-4 self-start">
                                    <i class="fa-solid fa-globe"></i> Napkorong 3D orientáció
                                </h3>
                                <canvas id="sunCanvas" width="280" height="280" class="rounded-xl border border-gray-800 bg-slate-950 shadow-inner"></canvas>
                                <p class="text-xs text-gray-400 text-center mt-3">
                                    A vörös szaggatott vonal jelzi a forgástengelyt (É-D pólusok dőlése).
                                </p>
                            </div>

                            <!-- Ephemeris Angles Panel -->
                            <div class="bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl space-y-4">
                                <h3 class="text-sm font-semibold tracking-wide text-gray-400 uppercase border-b border-gray-800 pb-2">
                                    <i class="fa-solid fa-compass"></i> Heliografikus adatok
                                </h3>
                                
                                <div class="space-y-3">
                                    <div class="p-3 bg-gray-950/50 rounded-xl border border-gray-800">
                                        <div class="flex justify-between items-center">
                                            <span class="text-gray-400 font-medium">P szög (Helyzeti szög):</span>
                                            <span id="valP" class="text-xl font-extrabold text-amber-400">--</span>
                                        </div>
                                        <p class="text-xs text-gray-500 mt-1">A forgástengely észak-déli iránya a geocentrikus északhoz képest.</p>
                                    </div>

                                    <div class="p-3 bg-gray-950/50 rounded-xl border border-gray-800">
                                        <div class="flex justify-between items-center">
                                            <span class="text-gray-400 font-medium">B0 (Heliofizikai szélesség):</span>
                                            <span id="valB0" class="text-xl font-extrabold text-orange-400">--</span>
                                        </div>
                                        <p class="text-xs text-gray-500 mt-1">A látható korong középpontjának szélessége a napiegyenlítőtől mérve.</p>
                                    </div>

                                    <div class="p-3 bg-gray-950/50 rounded-xl border border-gray-800">
                                        <div class="flex justify-between items-center">
                                            <span class="text-gray-400 font-medium">L0 (Carrington hosszúság):</span>
                                            <span id="valL0" class="text-xl font-extrabold text-rose-400">--</span>
                                        </div>
                                        <p class="text-xs text-gray-500 mt-1">A látható korong középpontjának aktuális Carrington-hosszúsága.</p>
                                    </div>
                                    
                                    <div class="p-3 bg-gray-950/50 rounded-xl border border-gray-800">
                                        <div class="flex justify-between items-center">
                                            <span class="text-gray-400 font-medium">CR (Carrington rotáció):</span>
                                            <span id="valCR" class="text-xl font-extrabold text-rose-400">--</span>
                                        </div>
                                        <p class="text-xs text-gray-500 mt-1">A látható korong középpontjának aktuális Carrington-hosszúsága.</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Lower: Sunrise/Sunset Events -->
                        <div class="bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl">
                            <h3 class="text-sm font-semibold tracking-wide text-gray-400 uppercase mb-4 border-b border-gray-800 pb-2">
                                <i class="fa-solid fa-clock"></i> Nap eseményei (Világidő / UT szerint)
                            </h3>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div class="bg-amber-950/20 border border-amber-500/20 rounded-xl p-4 text-center">
                                    <i class="fa-solid fa-cloud-sun text-2xl text-amber-400 mb-1"></i>
                                    <div class="text-xs text-gray-400">Napkelte</div>
                                    <div id="eventSunrise" class="text-2xl font-black text-amber-300 mt-1">--</div>
                                </div>

                                <div class="bg-orange-950/20 border border-orange-500/20 rounded-xl p-4 text-center">
                                    <i class="fa-solid fa-arrows-up-to-line text-2xl text-orange-400 mb-1"></i>
                                    <div class="text-xs text-gray-400">Delelés (Solar Noon)</div>
                                    <div id="eventTransit" class="text-2xl font-black text-orange-300 mt-1">--</div>
                                </div>

                                <div class="bg-rose-950/20 border border-rose-500/20 rounded-xl p-4 text-center">
                                    <i class="fa-solid fa-cloud-moon text-2xl text-rose-400 mb-1"></i>
                                    <div class="text-xs text-gray-400">Napnyugta</div>
                                    <div id="eventSunset" class="text-2xl font-black text-rose-300 mt-1">--</div>
                                </div>
                            </div>
                        </div>

                        <!-- JSON Raw Output for Developers -->
                        <div class="bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl space-y-3">
                            <div class="flex justify-between items-center border-b border-gray-800 pb-2">
                                <h3 class="text-sm font-semibold tracking-wide text-gray-400 uppercase">
                                    <i class="fa-solid fa-code"></i> API Válasz (JSON)
                                </h3>
                                <button onclick="copyJson()" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded transition duration-150 flex items-center gap-1">
                                    <i class="fa-solid fa-copy"></i> Másolás
                                </button>
                            </div>
                            <pre id="rawJson" class="text-xs font-mono bg-black/60 p-4 rounded-xl border border-gray-850 overflow-x-auto text-emerald-400 max-h-64"></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Set default date to today in local timezone
            const today = new Date();
            const year = today.getFullYear();
            const month = String(today.getMonth() + 1).padStart(2, '0');
            const day = String(today.getDate()).padStart(2, '0');
            document.getElementById('inputDate').value = `${year}-${month}-${day}`;

            // Set default time to 12:00:00
            document.getElementById('inputTime').value = "12:00:00";

            // Drawing the Sun and its Grid
            function drawSunDisk(p, b0, l0) {
                const canvas = document.getElementById('sunCanvas');
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const w = canvas.width;
                const h = canvas.height;
                const cx = w / 2;
                const cy = h / 2;
                const r = w * 0.38; // Radius on canvas

                // Deep space canvas backdrop
                ctx.fillStyle = '#05070d';
                ctx.fillRect(0, 0, w, h);

                // Tiny stars background
                ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
                for (let i = 0; i < 25; i++) {
                    let sx = (i * 149) % w;
                    let sy = (i * 491) % h;
                    ctx.fillRect(sx, sy, 1, 1);
                }

                // Solar radial light gradient
                const radGrad = ctx.createRadialGradient(cx - r*0.15, cy - r*0.15, r*0.1, cx, cy, r);
                radGrad.addColorStop(0, '#fffbeb');
                radGrad.addColorStop(0.15, '#fef08a');
                radGrad.addColorStop(0.55, '#f97316');
                radGrad.addColorStop(1, '#7c2d12');

                ctx.save();
                
                // Draw the sun disk sphere
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, 2 * Math.PI);
                ctx.fillStyle = radGrad;
                ctx.fill();
                
                // Outer corona glow
                ctx.shadowColor = '#f97316';
                ctx.shadowBlur = 15;
                ctx.strokeStyle = '#fed7aa';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                
                ctx.shadowBlur = 0; // Reset shadow

                // Apply rotational P angle
                // P is measured counterclockwise, canvas rotates clockwise, hence -P
                ctx.translate(cx, cy);
                ctx.rotate(-p * Math.PI / 180);

                // Heliographic coordinate grid parameters
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                ctx.lineWidth = 1;

                const b0Rad = b0 * Math.PI / 180;
                
                // Draw latitude circles
                function drawParallel(phi_deg, isEquator = false) {
                    const phi = phi_deg * Math.PI / 180;
                    ctx.beginPath();
                    if (isEquator) {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                        ctx.lineWidth = 1.5;
                    } else {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                        ctx.lineWidth = 0.8;
                    }

                    let first = true;
                    for (let theta_deg = -90; theta_deg <= 90; theta_deg += 2) {
                        const theta = theta_deg * Math.PI / 180;
                        const x = r * Math.cos(phi) * Math.sin(theta);
                        const y = -(r * Math.sin(phi) * Math.cos(b0Rad) - r * Math.cos(phi) * Math.sin(b0Rad) * Math.cos(theta));
                        
                        if (x*x + y*y <= r*r + 0.5) {
                            if (first) {
                                ctx.moveTo(x, y);
                                first = false;
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                    }
                    ctx.stroke();
                }

                // Draw longitude meridians relative to central meridian
                function drawMeridian(theta_deg, isCentral = false) {
                    const theta = theta_deg * Math.PI / 180;
                    ctx.beginPath();
                    if (isCentral) {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                        ctx.lineWidth = 1.5;
                    } else {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                        ctx.lineWidth = 0.8;
                    }

                    let first = true;
                    for (let phi_deg = -90; phi_deg <= 90; phi_deg += 2) {
                        const phi = phi_deg * Math.PI / 180;
                        const x = r * Math.cos(phi) * Math.sin(theta);
                        const y = -(r * Math.sin(phi) * Math.cos(b0Rad) - r * Math.cos(phi) * Math.sin(b0Rad) * Math.cos(theta));
                        
                        if (x*x + y*y <= r*r + 0.5) {
                            if (first) {
                                ctx.moveTo(x, y);
                                first = false;
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                    }
                    ctx.stroke();
                }

                // Draw Coordinate Web
                drawParallel(0, true); // Equator
                drawParallel(30);
                drawParallel(-30);
                drawParallel(60);
                drawParallel(-60);

                drawMeridian(0, true); // Central Meridian
                drawMeridian(30);
                drawMeridian(-30);
                drawMeridian(60);
                drawMeridian(-60);

                // Draw Rotational Axis
                ctx.beginPath();
                ctx.moveTo(0, -r - 10);
                ctx.lineTo(0, r + 10);
                ctx.strokeStyle = '#f43f5e';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Label Axis Poles
                ctx.fillStyle = '#f43f5e';
                ctx.font = 'bold 11px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('N', 0, -r - 15);
                ctx.fillText('S', 0, r + 22);

                ctx.restore(); // Undo rotational transform
                
                // Draw stable vertical north pointer for reference
                ctx.beginPath();
                ctx.moveTo(cx, cy - r - 22);
                ctx.lineTo(cx, cy - r - 32);
                ctx.strokeStyle = '#9ca3af';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                ctx.fillStyle = '#9ca3af';
                ctx.font = '9px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Geocentrikus Észak', cx, cy - r - 35);
            }

            let lastResponseData = null;

            function copyJson() {
                if (!lastResponseData) return;
                navigator.clipboard.writeText(JSON.stringify(lastResponseData, null, 2));
                alert("JSON válasz kimásolva a vágólapra!");
            }

            // Async call API to calculate solar ephemeris
            async function runCalculation() {
                const dateVal = document.getElementById('inputDate').value;
                const timeVal = document.getElementById('inputTime').value;
                const latVal = document.getElementById('inputLat').value;
                const lonVal = document.getElementById('inputLon').value;

                const loading = document.getElementById('loading');
                const errorBox = document.getElementById('errorBox');
                const results = document.getElementById('results');

                loading.classList.remove('hidden');
                errorBox.classList.add('hidden');

                // Build query url
                const url = `/api/ephemeris?date=${dateVal}&time=${timeVal}&lat=${latVal}&lon=${lonVal}`;

                try {
                    const response = await fetch(url);
                    if (!response.ok) {
                        const errData = await response.json();
                        throw new Error(errData.detail || "Ismeretlen API hiba lépett fel.");
                    }
                    
                    const data = await response.json();
                    lastResponseData = data;

                    // Update UI cards
                    document.getElementById('valP').innerText = data.efemeriszek.P_formazott;
                    document.getElementById('valB0').innerText = data.efemeriszek.B0_formazott;
                    document.getElementById('valL0').innerText = data.efemeriszek.L0_formazott;
                    document.getElementById('valCR').innerText = data.efemeriszek.CR_formazott;

                    document.getElementById('eventSunrise').innerText = data.nap_esemenyek_ut.napkelte;
                    document.getElementById('eventTransit').innerText = data.nap_esemenyek_ut.deletes_del;
                    document.getElementById('eventSunset').innerText = data.nap_esemenyek_ut.napnyugta;

                    // Format and render raw JSON
                    document.getElementById('rawJson').innerText = JSON.stringify(data, null, 2);

                    // Redraw graphics canvas
                    drawSunDisk(data.efemeriszek.P, data.efemeriszek.B0, data.efemeriszek.L0);

                } catch (err) {
                    document.getElementById('errorMessage').innerText = err.message;
                    errorBox.classList.remove('hidden');
                } finally {
                    loading.classList.add('hidden');
                }
            }

            // Listen to form submit
            document.getElementById('calcForm').addEventListener('submit', function(e) {
                e.preventDefault();
                runCalculation();
            });

            // Initial auto-run
            window.addEventListener('load', runCalculation);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # In production/deployment, PORT environment variable is automatically assigned
    port = int(os.environ.get("PORT", 8000))
    #uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    uvicorn.run("sun_ephemeris_web_api:app", host="0.0.0.0", port=port, reload=True)