#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"image.h"
#include <sys/time.h>
#include <omp.h>
#define WINDOW 3
#define THRESH 40
#define STEP 0.25f
#define CONST 150.0f
#define ITERS 25
#define PASSES 1

#define ROWBLOCK 8
#define COLBLOCK 8


int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

template<class T>
float PSNR(T *img1, T *img2, int width, int height)
{
        float psnr;
        float MSE =0.0f;
        #pragma omp parallel for private(psnr)
        for(int i=0; i <height; i++)
        {
                for(int j=0;j<width;j++)
                {
                        float err=(img1[(i+1)*(width+2)+(j+1)]-img2[(i+1)*(width+2)+(j+1)]);
                        MSE = MSE+err*err;
                }
        }

        MSE=MSE/(height*width);
        psnr= 10 * (float)log10(255.0*255.0/MSE);
        return(psnr);
}
	
 void mediankernel(int *ip_img, int *op_img, int height,int width)
{
	int i,j,k,m;
	int center_value;
	int array[9],counter;
    #pragma omp parallel for private(counter, center_value, i, j, k, m, array)
	for(i=0;i<height;i++)
	{
	  for(j=0;j<width;j++) {
		counter=0;
		
		for(k=-1;k<=1;k++)
			for(m=-1;m<=1;m++)
					array[counter++]=ip_img[(i+k+1)*(width+2)+(j+m+1)];
		center_value=array[4];
	  	qsort(array,9,sizeof(int),cmpfunc);	 	
	
        	op_img[(i+1)*(width+2)+(j+1)]=(int)(abs(center_value-array[4])>THRESH);
           }
        }
}

void filterPDE_kernel(int *ip_img, int *med_img, int *op_img,int height,int width)
{
	
        int i,j;

	int pn,ps,pe,pw,pc;
	int mn,ms,me,mw;
        int gn,gs, ge, gw;
        float cn, cs, ce, cw;
	
    //#pragma omp parallel for private(pn, ps, pe, pw, pc, mn, ms, me, mw, gn, gs, ge, gw, cn, cs, ce, cw)	
	for(i=0;i<height;i++) {
	  for(j=0;j<width;j++) {
	//Fetch Pixels
	//Input Image
		pn=ip_img[(i-1+1)*(width+2)+(j+0+1)];
		ps=ip_img[(i+1+1)*(width+2)+(j+0+1)];
		pe=ip_img[(i+0+1)*(width+2)+(j-1+1)];
		pw=ip_img[(i+0+1)*(width+2)+(j+1+1)];
		pc=ip_img[(i+0+1)*(width+2)+(j+0+1)];
	//Median Pixels
		mn=med_img[(i-1+1)*(width+2)+(j+0+1)];
		ms=med_img[(i+1+1)*(width+2)+(j+0+1)];
		me=med_img[(i+0+1)*(width+2)+(j-1+1)];
		mw=med_img[(i+0+1)*(width+2)+(j+1+1)];
		//mc=med_img[(i+0+1)*(width+2)+(j+0+1)];

     	   gn=(pn-pc)*(int)(mn==0.0);
     	   gs=(ps-pc)*(int)(ms==0.0);
      	  ge=(pe-pc)*(int)(me==0.0);
       	 gw=(pw-pc)*(int)(mw==0.0);

       	 cn=(CONST*CONST)/(CONST*CONST+gn*gn);
       	 cs=(CONST*CONST)/(CONST*CONST+gs*gs);
       	 ce=(CONST*CONST)/(CONST*CONST+ge*ge);
       	 cw=(CONST*CONST)/(CONST*CONST+gw*gw);

	op_img[(i+1)*(width+2)+(j+1)]=pc+(int)(STEP*(cn*gn+cs*gs+ce*ge+cw*gw));
	}
   }
}



int main(int argc, char **argv)
{

	//Variables required throughout the code
	int width,height;
	int *org_img, *noised_img, *op_img, *tmp_ip_img,*tmp_img;
	float psnr;
	struct timeval start,end;
	if(argc<4)
	{
		printf("No image file specified!\nExiting...\n");
		printf("\n do: ./adf <image> <noised image> <threads>\n");
		exit(0);
	}

    omp_set_num_threads(atoi(argv[3]));

	//Load Original Image
	read_image_template<int>(argv[1],&org_img,&width,&height);
	add_padding<int>(&org_img,width,height);
	
	//Load Noised Image
	read_image_template<int>(argv[2],&noised_img,&width,&height);
	add_padding<int>(&noised_img,width,height);
	
	//Output Image
	
	op_img = (int *)malloc(sizeof(int)*(width+2)*(height+2));
	tmp_ip_img = (int *)malloc(sizeof(int)*(width+2)*(height+2));
	tmp_img = (int *)malloc(sizeof(int)*(width+2)*(height+2));


	gettimeofday(&start,NULL);

	//Calculate PSNR
	psnr=PSNR<int>(org_img,noised_img,width,height);
	//printf("Original PSNR=%f\n",psnr);

	for(int k=0; k<PASSES; k++)
	{
		
		memcpy(tmp_ip_img,noised_img,sizeof(int)*(width+2)*(height+2));

		for(int i=0;i<ITERS;i++)
		{

			//Median Kernel 
		
			mediankernel (tmp_ip_img, tmp_img,height,width);

			
			filterPDE_kernel (tmp_ip_img,tmp_img,op_img,height,width);	
			
			#if 1
			int *tmp_ptr=op_img;
			op_img=tmp_ip_img;
			tmp_ip_img=tmp_ptr;				
			#endif
		}
		
	
		psnr=PSNR<int>(org_img,op_img,width,height);

	}

	gettimeofday(&end,NULL);

	printf("\n %d %d: %f\n",height,atoi(argv[3]),(float)((end.tv_sec*1000+end.tv_usec/1000)-(start.tv_sec*1000+start.tv_usec/1000)));

	//Write the image
	remove_padding<int>(&op_img,width,height);
	char name[]="op.pgm";
	write_image_template<int>(name,op_img,width,height);
	
	//Clean up	
	free(org_img);
	free(noised_img);
	free(op_img);
	free(tmp_ip_img);
	free(tmp_img);
			
	return 0; 
}
