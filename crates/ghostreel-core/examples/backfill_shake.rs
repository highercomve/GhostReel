//! Measure steadiness for videos that were indexed before the measurement existed.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = ghostreel_core::paths::Paths::resolve()?;
    let config = ghostreel_core::config::Config::load(&paths.config_file).unwrap_or_default();
    let db = ghostreel_core::db::Db::open(&paths.db_file())?;
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").ok_or("ffmpeg not found")?;
    let args: Vec<i64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (limit, only): (usize, Vec<i64>) =
        if args.len() == 1 { (args[0] as usize, vec![]) } else { (usize::MAX, args) };

    let mut st = db.conn.prepare(
        "SELECT v.id, v.duration_s, vf.path FROM videos v JOIN video_files vf ON vf.video_id = v.id
         WHERE v.duration_s IS NOT NULL GROUP BY v.id ORDER BY v.id",
    )?;
    let rows: Vec<(i64, f64, String)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(Result::ok)
        .filter(|(id, _, _)| only.is_empty() || only.contains(id))
        .filter(|(id, _, _)| !only.is_empty() || db.motion_windows(*id).map(|w| w.is_empty()).unwrap_or(true))
        .take(limit)
        .collect();

    for (id, duration, path) in rows {
        let w = ghostreel_core::steadiness::measure(
            &ffmpeg,
            std::path::Path::new(&path),
            duration,
            config.script.shake_window_s,
            config.script.shake_stride_s,
            |_| {},
        )
        .await?;
        let worst = w.iter().map(|x| x.jerk).fold(0.0f64, f64::max);
        let mut sorted: Vec<f64> = w.iter().map(|x| x.jerk).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        db.set_motion_windows(id, &w)?;
        let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy();
        let verdict = if worst > config.script.max_shake_jerk { "SHAKY" } else { "steady" };
        println!("#{id:<4} {name:32} windows={:<3} median={median:.2} worst={worst:.2} {verdict}", w.len());
    }
    Ok(())
}
