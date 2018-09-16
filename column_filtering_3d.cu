/********************
*
* CUDA Kernel: column gradient computing
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
void column_filtering(double * R, 
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
    int page=j/n;
	int col=j-page*n;    
        R[sub2ind(i,j,m)] = M[sub2ind(i,min(col+1,n-1)+page*n,m)]-M[sub2ind(i,j,m)];
        return ;
}