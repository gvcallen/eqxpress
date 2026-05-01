import jax

def tree_op(op_fn):
    """Wraps an AbstractExpression to intelligently map across PyTrees and broadcast scalars."""
    def wrapped(*args):
        if len(args) == 1:
            # Unary operations (e.g., -op)
            return jax.tree_util.tree_map(op_fn, args[0])
        
        elif len(args) == 2:
            # Binary operations (e.g., op + op, op * scalar)
            left, right = args
            
            # In JAX, a single scalar or array has exactly 1 node in its tree structure
            left_is_leaf = jax.tree_util.tree_structure(left).num_nodes == 1
            right_is_leaf = jax.tree_util.tree_structure(right).num_nodes == 1
            
            if left_is_leaf and not right_is_leaf:
                # Left is a scalar/array, broadcast it over the right PyTree
                return jax.tree_util.tree_map(lambda r: op_fn(left, r), right)
            
            elif right_is_leaf and not left_is_leaf:
                # Right is a scalar/array, broadcast it over the left PyTree
                return jax.tree_util.tree_map(lambda l: op_fn(l, right), left)
            
            else:
                # Both are trees (must have matching structures) or both are scalars
                return jax.tree_util.tree_map(op_fn, left, right)
        else:
            raise ValueError("Only unary and binary AbstractExpressions are supported.")
            
    return wrapped