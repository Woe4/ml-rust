use crate::{Point, KMeans};
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BaggedKMeans {
    n_estimators: usize,
    k: usize,
    max_iterations: usize,
    tolerance: f64,
    sample_size: Option<usize>,
}

#[derive(Debug)]
pub struct BaggedClusterResult {
    pub consensus_centroids: Vec<Point>,
    pub consensus_assignments: Vec<usize>,
    pub individual_results: Vec<(Vec<Point>, Vec<usize>)>,
    pub confidence_scores: Vec<f64>,
}

impl BaggedKMeans {
    /// Create a new BaggedKMeans instance
    /// 
    /// # Arguments
    /// * `n_estimators` - Number of k-means models to train
    /// * `k` - Number of clusters for each k-means model
    /// * `max_iterations` - Maximum iterations per k-means model
    /// * `tolerance` - Convergence tolerance for each model
    /// * `sample_size` - Size of bootstrap samples (None = same as original data size)
    pub fn new(
        n_estimators: usize,
        k: usize,
        max_iterations: usize,
        tolerance: f64,
        sample_size: Option<usize>,
    ) -> Self {
        BaggedKMeans {
            n_estimators,
            k,
            max_iterations,
            tolerance,
            sample_size,
        }
    }

    /// Create a bootstrap sample from the original data
    fn create_bootstrap_sample(&self, data: &[Point]) -> Vec<Point> {
        let mut rng = rand::thread_rng();
        let sample_size = self.sample_size.unwrap_or(data.len());
        
        (0..sample_size)
            .map(|_| data[rng.gen_range(0..data.len())])
            .collect()
    }

    /// Train multiple k-means models on bootstrap samples
    pub fn fit(&self, data: &[Point]) -> Result<BaggedClusterResult, String> {
        if data.is_empty() {
            return Err("Cannot cluster empty data".to_string());
        }

        if self.n_estimators == 0 {
            return Err("Number of estimators must be greater than 0".to_string());
        }

        let mut individual_results = Vec::new();
        println!("Training {} k-means models on bootstrap samples...", self.n_estimators);

        // Train individual k-means models
        for i in 0..self.n_estimators {
            let bootstrap_sample = self.create_bootstrap_sample(data);
            let kmeans = KMeans::new(self.k, self.max_iterations, self.tolerance);
            
            match kmeans.fit(&bootstrap_sample) {
                Ok((centroids, assignments)) => {
                    // Map bootstrap assignments back to original data indices
                    let full_assignments = self.map_assignments_to_original(data, &bootstrap_sample, &assignments, &centroids);
                    individual_results.push((centroids, full_assignments));
                    println!("Model {} completed", i + 1);
                }
                Err(e) => {
                    return Err(format!("Error in estimator {}: {}", i + 1, e));
                }
            }
        }

        // Create consensus clustering
        let consensus_result = self.create_consensus_clustering(data, &individual_results)?;
        
        Ok(consensus_result)
    }

    /// Map bootstrap sample assignments back to original data
    fn map_assignments_to_original(
        &self,
        original_data: &[Point],
        _bootstrap_sample: &[Point],
        _bootstrap_assignments: &[usize],
        centroids: &[Point],
    ) -> Vec<usize> {
        original_data
            .iter()
            .map(|point| {
                // Find the nearest centroid for each original point
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

    /// Create consensus clustering from individual results
    fn create_consensus_clustering(
        &self,
        data: &[Point],
        individual_results: &[(Vec<Point>, Vec<usize>)],
    ) -> Result<BaggedClusterResult, String> {
        let n_points = data.len();
        let mut consensus_assignments = vec![0; n_points];
        let mut confidence_scores = vec![0.0; n_points];

        // For each point, determine consensus cluster assignment
        for point_idx in 0..n_points {
            let mut cluster_votes: HashMap<usize, usize> = HashMap::new();
            
            // Collect votes from all models
            for (_, assignments) in individual_results {
                let cluster = assignments[point_idx];
                *cluster_votes.entry(cluster).or_insert(0) += 1;
            }

            // Find the cluster with the most votes
            let (consensus_cluster, vote_count) = cluster_votes
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .unwrap_or((0, 0));

            consensus_assignments[point_idx] = consensus_cluster;
            confidence_scores[point_idx] = vote_count as f64 / self.n_estimators as f64;
        }

        // Calculate consensus centroids
        let consensus_centroids = self.calculate_consensus_centroids(data, &consensus_assignments);

        Ok(BaggedClusterResult {
            consensus_centroids,
            consensus_assignments,
            individual_results: individual_results.to_vec(),
            confidence_scores,
        })
    }

    /// Calculate centroids based on consensus assignments
    fn calculate_consensus_centroids(&self, data: &[Point], assignments: &[usize]) -> Vec<Point> {
        let mut clusters: HashMap<usize, Vec<Point>> = HashMap::new();
        
        // Group points by consensus cluster assignment
        for (point, &cluster_id) in data.iter().zip(assignments) {
            clusters.entry(cluster_id).or_insert_with(Vec::new).push(*point);
        }

        // Calculate centroids for each cluster
        (0..self.k)
            .map(|i| {
                if let Some(cluster_points) = clusters.get(&i) {
                    Point::centroid(cluster_points)
                } else {
                    Point::new(0.0, 0.0)
                }
            })
            .collect()
    }

    /// Calculate stability score based on agreement between models
    pub fn calculate_stability(&self, result: &BaggedClusterResult) -> f64 {
        let mean_confidence: f64 = result.confidence_scores.iter().sum::<f64>() / result.confidence_scores.len() as f64;
        mean_confidence
    }

    /// Get cluster quality metrics
    pub fn evaluate_clustering(&self, data: &[Point], result: &BaggedClusterResult) -> ClusteringMetrics {
        let inertia = self.calculate_inertia(data, &result.consensus_centroids, &result.consensus_assignments);
        let stability = self.calculate_stability(result);
        let silhouette_score = self.calculate_silhouette_score(data, &result.consensus_assignments, &result.consensus_centroids);

        ClusteringMetrics {
            inertia,
            stability,
            silhouette_score,
            n_models: self.n_estimators,
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, data: &[Point], centroids: &[Point], assignments: &[usize]) -> f64 {
        data.iter()
            .zip(assignments.iter())
            .map(|(point, &cluster_id)| {
                if cluster_id < centroids.len() {
                    point.distance_to(&centroids[cluster_id]).powi(2)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Calculate simplified silhouette score
    fn calculate_silhouette_score(&self, data: &[Point], assignments: &[usize], centroids: &[Point]) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mut total_silhouette = 0.0;
        let mut count = 0;

        for (i, point) in data.iter().enumerate() {
            let own_cluster = assignments[i];
            
            // Calculate average distance to points in same cluster (a)
            let same_cluster_points: Vec<&Point> = data.iter()
                .enumerate()
                .filter(|(j, _)| *j != i && assignments[*j] == own_cluster)
                .map(|(_, p)| p)
                .collect();

            let a = if same_cluster_points.is_empty() {
                0.0
            } else {
                same_cluster_points.iter()
                    .map(|p| point.distance_to(p))
                    .sum::<f64>() / same_cluster_points.len() as f64
            };

            // Calculate minimum average distance to points in other clusters (b)
            let mut min_b = f64::INFINITY;
            for cluster_id in 0..self.k {
                if cluster_id != own_cluster {
                    let other_cluster_points: Vec<&Point> = data.iter()
                        .enumerate()
                        .filter(|(j, _)| assignments[*j] == cluster_id)
                        .map(|(_, p)| p)
                        .collect();

                    if !other_cluster_points.is_empty() {
                        let avg_dist = other_cluster_points.iter()
                            .map(|p| point.distance_to(p))
                            .sum::<f64>() / other_cluster_points.len() as f64;
                        min_b = min_b.min(avg_dist);
                    }
                }
            }

            if min_b != f64::INFINITY {
                let silhouette = (min_b - a) / a.max(min_b);
                total_silhouette += silhouette;
                count += 1;
            }
        }

        if count > 0 {
            total_silhouette / count as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct ClusteringMetrics {
    pub inertia: f64,
    pub stability: f64,
    pub silhouette_score: f64,
    pub n_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bagged_kmeans_creation() {
        let bagged_kmeans = BaggedKMeans::new(5, 3, 100, 0.01, None);
        assert_eq!(bagged_kmeans.n_estimators, 5);
        assert_eq!(bagged_kmeans.k, 3);
    }

    #[test]
    fn test_bootstrap_sampling() {
        let data = vec![
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ];
        let bagged_kmeans = BaggedKMeans::new(3, 2, 50, 0.01, Some(3));
        let sample = bagged_kmeans.create_bootstrap_sample(&data);
        assert_eq!(sample.len(), 3);
    }
}