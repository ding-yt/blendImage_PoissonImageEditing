/********************
*
* CUDA Kernel: row gradient computing
*
*/


/* ==================================================
 *
 * sub2ind - Column-major indexing of 2D arrays
 *
 */
template <typename T>
__device__ __forceinline__ T sub2ind( T i, T j, T height) {
  return (i + height*j);

}  // end function 'sub2ind'


/* ==================================================
 *
 *  core kernel
 *
 */

__global__ 
void row_filtering(double * R, 
	const double * M,
	const int m, 
	const int n, 
	const int p){

/* thread indices */
        const int j = blockIdx.y*blockDim.y+threadIdx.y;
        const int i = blockIdx.x*blockDim.x+threadIdx.x;
        
/* matrix calculation */
	if ((i >= m) || (j >= n*p)  ){
                return;
        }
       
        R[sub2ind(i,j,m)] = M[sub2ind(min(i+1,m-1),j,m)]-M[sub2ind(i,j,m)];
        return ;
}