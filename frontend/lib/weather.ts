/**
 * Detects current weather using browser geolocation + Open-Meteo API.
 * No API key required.
 */

export type Weather = 'clear' | 'rain'

// WMO weather codes that indicate precipitation
// https://open-meteo.com/en/docs#weathervariables
const RAIN_CODES = new Set([
  51, 53, 55,       // Drizzle
  56, 57,           // Freezing drizzle
  61, 63, 65,       // Rain
  66, 67,           // Freezing rain
  71, 73, 75, 77,   // Snow (treat as rain for animation purposes)
  80, 81, 82,       // Rain showers
  85, 86,           // Snow showers
  95, 96, 99,       // Thunderstorm
])

async function getCoords(): Promise<{ lat: number; lon: number }> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation not supported'))
      return
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      (err) => reject(err),
      { timeout: 5000 }
    )
  })
}

export async function fetchWeather(): Promise<Weather> {
  try {
    const { lat, lon } = await getCoords()
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=precipitation,weathercode&timezone=auto`
    const res  = await fetch(url, { signal: AbortSignal.timeout(6000) })
    const data = await res.json()

    const code          = data?.current?.weathercode ?? 0
    const precipitation = data?.current?.precipitation ?? 0

    return (precipitation > 0 || RAIN_CODES.has(code)) ? 'rain' : 'clear'
  } catch {
    // Fail silently — default to clear if geolocation denied or network down
    return 'clear'
  }
}
