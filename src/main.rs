use rand::Rng;
use std::collections::HashMap;

mod bagging;
use bagging::{BaggedKMeans, ClusteringMetrics};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    // Calculate Euclidean distance to another point
    pub fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    // Calculate the centroid of a collection of points
    pub fn centroid(points: &[Point]) -> Point {
        if points.is_empty() {
            return Point::new(0.0, 0.0);
        }
        
        let sum_x: f64 = points.iter().map(|p| p.x).sum();
        let sum_y: f64 = points.iter().map(|p| p.y).sum();
        let count = points.len() as f64;
        
        Point::new(sum_x / count, sum_y / count)
    }
}

pub struct KMeans {
    k: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl KMeans {
    pub fn new(k: usize, max_iterations: usize, tolerance: f64) -> Self {
        KMeans {
            k,
            max_iterations,
            tolerance,
        }
    }

    // Initialize centroids randomly within the bounds of the data
    fn initialize_centroids(&self, data: &[Point]) -> Vec<Point> {
        if data.is_empty() {
            return vec![Point::new(0.0, 0.0); self.k];
        }

        let mut rng = rand::thread_rng();
        
        // Find bounds of the data
        let min_x = data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max_x = data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = data.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = data.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        (0..self.k)
            .map(|_| {
                Point::new(
                    rng.gen_range(min_x..=max_x),
                    rng.gen_range(min_y..=max_y),
                )
            })
            .collect()
    }

    // Assign each point to the nearest centroid
    fn assign_to_clusters(&self, data: &[Point], centroids: &[Point]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        point.distance_to(a).partial_cmp(&point.distance_to(b)).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    // Update centroids based on assigned points
    fn update_centroids(&self, data: &[Point], assignments: &[usize]) -> Vec<Point> {
        let mut clusters: HashMap<usize, Vec<Point>> = HashMap::new();
        
        // Group points by cluster assignment
        for (point, &cluster_id) in data.iter().zip(assignments) {
            clusters.entry(cluster_id).or_insert_with(Vec::new).push(*point);
        }

        // Calculate new centroids
        (0..self.k)
            .map(|i| {
                if let Some(cluster_points) = clusters.get(&i) {
                    Point::centroid(cluster_points)
                } else {
                    // If no points assigned to this centroid, keep it where it is
                    Point::new(0.0, 0.0) // This will be handled by the calling code
                }
            })
            .collect()
    }

    // Check if centroids have converged
    fn has_converged(&self, old_centroids: &[Point], new_centroids: &[Point]) -> bool {
        old_centroids
            .iter()
            .zip(new_centroids.iter())
            .all(|(old, new)| old.distance_to(new) < self.tolerance)
    }

    // Main k-means clustering algorithm
    pub fn fit(&self, data: &[Point]) -> Result<(Vec<Point>, Vec<usize>), String> {
        if data.is_empty() {
            return Err("Cannot cluster empty data".to_string());
        }
        
        if self.k == 0 {
            return Err("Number of clusters (k) must be greater than 0".to_string());
        }

        if self.k > data.len() {
            return Err("Number of clusters cannot exceed number of data points".to_string());
        }

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data);
        let mut assignments = vec![0; data.len()];

        for iteration in 0..self.max_iterations {
            // Assign points to nearest centroids
            assignments = self.assign_to_clusters(data, &centroids);

            // Update centroids
            let new_centroids = self.update_centroids(data, &assignments);

            // Check for convergence
            if self.has_converged(&centroids, &new_centroids) {
                println!("Converged after {} iterations", iteration + 1);
                centroids = new_centroids;
                break;
            }

            centroids = new_centroids;

            if iteration == self.max_iterations - 1 {
                println!("Reached maximum iterations ({})", self.max_iterations);
            }
        }

        Ok((centroids, assignments))
    }

    // Calculate the sum of squared distances (inertia)
    pub fn calculate_inertia(&self, data: &[Point], centroids: &[Point], assignments: &[usize]) -> f64 {
        data.iter()
            .zip(assignments.iter())
            .map(|(point, &cluster_id)| point.distance_to(&centroids[cluster_id]).powi(2))
            .sum()
    }
}

// Helper function to generate sample data
fn generate_sample_data() -> Vec<Point> {
    vec![
        // Cluster 1 (around 2, 2)
        Point::new(1.5, 1.8),
        Point::new(2.1, 2.3),
        Point::new(1.8, 2.1),
        Point::new(2.3, 1.9),
        Point::new(1.9, 2.0),
        
        // Cluster 2 (around 8, 8)
        Point::new(7.8, 8.1),
        Point::new(8.2, 7.9),
        Point::new(8.1, 8.3),
        Point::new(7.9, 8.0),
        Point::new(8.0, 8.2),
        
        // Cluster 3 (around 2, 8)
        Point::new(1.9, 8.1),
        Point::new(2.1, 7.9),
        Point::new(2.0, 8.2),
        Point::new(1.8, 8.0),
        Point::new(2.2, 8.1),
        
        // Some outliers
        Point::new(5.0, 5.0),
        Point::new(6.0, 3.0),
        Point::new(3.5, 6.0),
    ]
}

fn main() {
    println!("K-Means Clustering Algorithm in Rust");
    println!("=====================================");

    // Generate sample data
    let data = generate_sample_data();
    
    println!("Sample data points:");
    for (i, point) in data.iter().enumerate() {
        println!("Point {}: ({:.2}, {:.2})", i + 1, point.x, point.y);
    }

    // Create k-means clustering with k=3
    let kmeans = KMeans::new(3, 100, 0.01);

    // Perform clustering
    match kmeans.fit(&data) {
        Ok((centroids, assignments)) => {
            println!("\nClustering Results:");
            println!("==================");
            
            // Print centroids
            println!("Final centroids:");
            for (i, centroid) in centroids.iter().enumerate() {
                println!("Cluster {}: ({:.2}, {:.2})", i + 1, centroid.x, centroid.y);
            }

            // Print assignments
            println!("\nPoint assignments:");
            for (i, (point, &cluster)) in data.iter().zip(&assignments).enumerate() {
                println!("Point {} ({:.2}, {:.2}) -> Cluster {}", 
                        i + 1, point.x, point.y, cluster + 1);
            }

            // Calculate and print inertia
            let inertia = kmeans.calculate_inertia(&data, &centroids, &assignments);
            println!("\nSum of squared distances (inertia): {:.2}", inertia);

            // Group points by cluster
            println!("\nClusters:");
            for cluster_id in 0..3 {
                println!("Cluster {}:", cluster_id + 1);
                for (i, (point, &assignment)) in data.iter().zip(&assignments).enumerate() {
                    if assignment == cluster_id {
                        println!("  Point {}: ({:.2}, {:.2})", i + 1, point.x, point.y);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error during clustering: {}", e);
        }
    }

    // Demonstrate different k values
    println!("\nTesting different k values:");
    println!("==========================");
    
    for k in 1..=5 {
        let kmeans_test = KMeans::new(k, 50, 0.01);
        if let Ok((centroids, assignments)) = kmeans_test.fit(&data) {
            let inertia = kmeans_test.calculate_inertia(&data, &centroids, &assignments);
            println!("k={}: inertia = {:.2}", k, inertia);
        }
    }

    // Demonstrate Bagged K-Means
    println!("\n{}", "=".repeat(50));
    println!("BAGGED K-MEANS CLUSTERING");
    println!("{}", "=".repeat(50));

    // Create bagged k-means with 10 estimators
    let bagged_kmeans = BaggedKMeans::new(
        10,     // n_estimators
        3,      // k clusters
        50,     // max_iterations per model
        0.01,   // tolerance
        None,   // sample_size (use full data size)
    );

    // Perform bagged clustering
    match bagged_kmeans.fit(&data) {
        Ok(bagged_result) => {
            println!("\nBagged K-Means Results:");
            println!("======================");

            // Print consensus centroids
            println!("Consensus centroids:");
            for (i, centroid) in bagged_result.consensus_centroids.iter().enumerate() {
                println!("Cluster {}: ({:.2}, {:.2})", i + 1, centroid.x, centroid.y);
            }

            // Print consensus assignments with confidence
            println!("\nConsensus assignments (with confidence):");
            for (i, (point, (&cluster, &confidence))) in data.iter()
                .zip(bagged_result.consensus_assignments.iter()
                .zip(bagged_result.confidence_scores.iter())).enumerate() {
                println!("Point {} ({:.2}, {:.2}) -> Cluster {} (confidence: {:.2})", 
                        i + 1, point.x, point.y, cluster + 1, confidence);
            }

            // Evaluate clustering quality
            let metrics = bagged_kmeans.evaluate_clustering(&data, &bagged_result);
            println!("\nClustering Quality Metrics:");
            println!("===========================");
            println!("Inertia: {:.2}", metrics.inertia);
            println!("Stability: {:.2}", metrics.stability);
            println!("Silhouette Score: {:.2}", metrics.silhouette_score);
            println!("Number of Models: {}", metrics.n_models);

            // Show points with low confidence (potential outliers/uncertain points)
            println!("\nLow Confidence Points (< 0.7):");
            println!("==============================");
            for (i, (point, (&cluster, &confidence))) in data.iter()
                .zip(bagged_result.consensus_assignments.iter()
                .zip(bagged_result.confidence_scores.iter())).enumerate() {
                if confidence < 0.7 {
                    println!("Point {} ({:.2}, {:.2}) -> Cluster {} (confidence: {:.2})", 
                            i + 1, point.x, point.y, cluster + 1, confidence);
                }
            }

            // Compare with single k-means
            println!("\nComparison with Single K-Means:");
            println!("===============================");
            let single_kmeans = KMeans::new(3, 50, 0.01);
            if let Ok((single_centroids, single_assignments)) = single_kmeans.fit(&data) {
                let single_inertia = single_kmeans.calculate_inertia(&data, &single_centroids, &single_assignments);
                println!("Single K-Means Inertia: {:.2}", single_inertia);
                println!("Bagged K-Means Inertia: {:.2}", metrics.inertia);
                println!("Improvement: {:.2}%", 
                    ((single_inertia - metrics.inertia) / single_inertia * 100.0).max(0.0));
            }
        }
        Err(e) => {
            eprintln!("Error during bagged clustering: {}", e);
        }
    }
}
