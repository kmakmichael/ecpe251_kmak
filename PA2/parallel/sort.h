//Source: Geeks for geeks

#include<stdlib.h>
#include<stdio.h>
#include <omp.h>
#define min(A,B) A>B?B:A
 
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(float arr[], int l, int m, int r);
void mergeSort(float arr[], int n,int threads)
{

   omp_set_num_threads(threads);

   int curr_size;  // For current size of subarrays to be merged
                   // curr_size varies from 1 to n/2
   int left_start; // For picking starting index of left subarray
                   //                       // to be merged
   int mid,right_end;
   for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
   {
	#pragma omp parallel for
	for (left_start=0; left_start<n-1; left_start += 2*curr_size)
       {
		 mid = min(left_start + curr_size - 1, n-1);
 
           right_end = min(left_start + 2*curr_size - 1, n-1);
	
		merge(arr, left_start, mid, right_end);
       }
   }
}

void merge(float arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    float *L, *R;

    L = (float *)malloc(sizeof(float)*n1);
    R = (float *)malloc(sizeof(float)*n2);
 
    /* Copy data to temp arrays L[] and R[] */
    //#pragma omp parallel for
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    //#pragma omp parallel for
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    
free(L);
free(R);
}

/* Function to print an array */
void printArray(float A[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%f ", A[i]);
    printf("\n");
}
