use nalgebra::{DMatrix, DVector};
use ptree::builder::TreeBuilder;
use ptree::output::print_tree;

#[derive(Default, Debug, Clone)]
pub struct RootedTreeVertex {
    partition: Vec<usize>,
    level: f64,
    level_diff: f64,
    sum: f64,
    left_child: Option<Box<RootedTreeVertex>>,
    right_child: Option<Box<RootedTreeVertex>>,
}

impl RootedTreeVertex {
    fn new(partition: Vec<usize>) -> RootedTreeVertex {
        RootedTreeVertex {
            partition: partition,
            ..Default::default()
        }
    }

    pub fn get_partition_tree(matrix: &DMatrix<f64>) -> Self {
        let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
        let mut root = RootedTreeVertex::new(vertex_ids);
        root.partition_tree_vertex(&matrix, 0.0);
        return root;
    }

    fn partition_tree_vertex(&mut self, matrix: &DMatrix<f64>, parent_level: f64) {
        let first_i = self.partition[0];
        if self.partition.len() == 1 {
            self.level_diff = matrix[(first_i, first_i)];
            self.level = parent_level;
            return;
        }
        let mut left_partition: Vec<usize> = vec![first_i];
        let mut right_partition: Vec<usize> = Vec::new();
        let mut min = f64::MAX;
        for &i in &self.partition[1..] {
            if min > matrix[(first_i, i)] {
                min = matrix[(first_i, i)];
                left_partition.extend(right_partition.iter());
                right_partition.clear();
                right_partition.push(i);
            } else {
                if min == matrix[(first_i, i)] {
                    right_partition.push(i);
                } else {
                    left_partition.push(i);
                }
            }
        }
        self.level = min;
        self.level_diff = min - parent_level;
        let left_child = Box::new(RootedTreeVertex::new(left_partition));
        let right_child = Box::new(RootedTreeVertex::new(right_partition));
        self.left_child = Some(left_child);
        self.right_child = Some(right_child);

        self.left_child
            .as_mut()
            .unwrap()
            .partition_tree_vertex(matrix, self.level);
        self.right_child
            .as_mut()
            .unwrap()
            .partition_tree_vertex(matrix, self.level);
    }

    pub fn multiply_with_tree(&mut self, vector: &DVector<f64>) -> DVector<f64> {
        self.calculate_sums(vector);
        let mut product: DVector<f64> = DVector::<f64>::zeros(vector.nrows());
        self.calculate_full_product(&mut product, 0.0);
        return product;
    }

    fn calculate_sums(&mut self, vector: &DVector<f64>) -> f64 {
        if self.partition.len() == 1 {
            let sum = vector[self.partition[0]];
            self.sum = (self.level_diff - self.level) * sum;
            return sum;
        }
        let left_sum = self.left_child.as_mut().unwrap().calculate_sums(vector);
        let right_sum = self.right_child.as_mut().unwrap().calculate_sums(vector);
        let sum = left_sum + right_sum;
        self.sum = sum * self.level_diff;
        return sum;
    }

    fn calculate_full_product(&self, product: &mut DVector<f64>, prev_sum: f64) {
        let sum = prev_sum + self.sum;
        if self.partition.len() == 1 {
            product[self.partition[0]] = sum;
            return;
        }
        self.left_child
            .as_ref()
            .unwrap()
            .calculate_full_product(product, sum);
        self.right_child
            .as_ref()
            .unwrap()
            .calculate_full_product(product, sum);
    }

    pub fn get_permutation_matrix(&self) -> DMatrix<f64> {
        let size = self.partition.len();
        let permutations = self.get_permutations();
        let mut perm_mat = DMatrix::<f64>::zeros(size, size);
        for (i, &j) in permutations.iter().enumerate() {
            perm_mat[(i, j)] = 1.;
        }
        return perm_mat;
    }
    fn get_permutations(&self) -> Vec<usize> {
        if self.partition.len() == 1 {
            return self.partition.clone();
        }
        let mut left = self.left_child.as_ref().unwrap().get_permutations();
        let right = self.right_child.as_ref().unwrap().get_permutations();
        left.extend(right.iter());
        return left;
    }
    pub fn print_rooted_tree(&self) {
        let mut tree_root = TreeBuilder::new(format!("{:?}", self.partition));
        self.left_child
            .as_ref()
            .unwrap()
            .construct_tree(&mut tree_root);
        self.left_child
            .as_ref()
            .unwrap()
            .construct_tree(&mut tree_root);
        let tree = tree_root.build();
        print_tree(&tree).ok();
    }
    fn construct_tree(&self, output_tree: &mut TreeBuilder) {
        if self.partition.len() == 1 {
            output_tree.add_empty_child(format!("{:?}", self.partition));
        } else {
            output_tree.begin_child(format!("{:?}", self.partition));
            self.left_child
                .as_ref()
                .unwrap()
                .construct_tree(output_tree);
            self.right_child
                .as_ref()
                .unwrap()
                .construct_tree(output_tree);
            output_tree.end_child();
        }
    }
}
