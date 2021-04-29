#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<omp.h>
#include<mpi.h>

int main(int argc, char *argv[])
{
	/***********Initialization*************/

	/*input variables of symbols, weight and currents and firings patterns*/

	if (argc != 3) {
		fprintf(stderr, "usage:  ./spiker  <image row size> <iter> ");
		exit(3);
	}
	int row, col;
	int iter;
	row = atoi(argv[1]);
	col = row;
	iter = atoi(argv[2]);

	int ORIG_IMG_SIZE = 24;
	//int TRAINING_SIZE = 48;
	int ro, co,nerve, ratio;
	float value;
	int Ni=48;//Training size=number of total images

    // MPI init
    int comm_size, comm_rank;
    int rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "MPI failed to init\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

	int Ne=row*col;//Total number of input neurons
	//int N=Ni+Ne;//total neurons
	float **input;
	float **w;
	int *L1_firings;
	float *level1_I, *level1_v, *level1_v0, *level1_w_variable;
	float *level2_I, *level2_v, *level2_v0, *level2_w_variable;
	L1_firings = (int *)malloc(sizeof(int *)*Ne);
	level1_I = (float *)malloc(sizeof(float *)*Ne);
	level1_v = (float *)malloc(sizeof(float *)*Ne);
	level1_v0 = (float *)malloc(sizeof(float *)*Ne);
	level1_w_variable = (float *)malloc(sizeof(float *)*Ne);
	level2_I = (float *)malloc(sizeof(float *)*Ni);
	level2_v = (float *)malloc(sizeof(float *)*Ni);
	level2_v0 = (float *)malloc(sizeof(float *)*Ni);
	level2_w_variable = (float *)malloc(sizeof(float *)*Ni);

	float start_clock;
	float end_clock;
	int loop;
	int index, orig_row, orig_col;

	int i,j,m;
	float t;

	input=(float **)malloc(sizeof(float *)*row);

	for (i=0;i<row;i++) {
		input[i] = (float *) malloc(sizeof(float)*col);
	}
	float * orig_input = (float *) malloc(sizeof(float)*24*24);
	w=(float **)malloc(sizeof(float *)*Ni);

	float * orig_w = (float *) malloc(sizeof(float)*24*24*Ni);

	for (i=0;i<Ni;i++) {
		w[i] = (float *) malloc(sizeof(float)*Ne);
	}
	/******firing time and firing neurons initialized*******/

	int L1_N_firings=1;

	for (m=0;m<Ne;m++) {
		L1_firings[m]=-5;//Neuronal index of  firing, initializing
	}

	/*******Reading test symbol from file*********/

	ratio = col/ORIG_IMG_SIZE;
	FILE *fs;
	fs=fopen("input-24by24by48.txt","r");//opening the file to read

	for (j=0;j<24*24;j++) {
		fscanf(fs,"%f",&orig_input[j]);//reading from I/p file
		orig_input[j]=orig_input[j]*5;
	}
	fclose(fs);
	FILE *fs1;
	fs1=fopen("weight-24by24by48.txt","r");//opening the file to read

	for (j=0;j<24*24*Ni;j++) {
		fscanf(fs1,"%f",&value);//reading from I/p file
		orig_w[j]=value/(ratio*ratio);//the division will be to reducing values
	}
	fclose(fs1);
	nerve = 0;
	float *new_image;

	float *new_weights;
	new_image = (float *) malloc(sizeof(float)*Ne);
	new_weights = (float *) malloc(sizeof(float)*Ne*Ni);
	orig_row = orig_col = ORIG_IMG_SIZE;
	for (i=0; i<Ne;i++ ) {
		ro = nerve/col;
		co = nerve%col;
		ro = ro/ratio;
		co = co/ratio;
		index = ro*ORIG_IMG_SIZE+co;
		new_image[i] = orig_input[index];
		nerve++;

		for(j=0;j<Ni;j++)
			new_weights[i*Ni+j] = orig_w[j*orig_row*orig_col+index];
	}
	for (i=0;i<row;i++) {//Number of symbols=no. of col
		for (j=0;j<row;j++) {
			input[i][j]= new_image[row*i+j];
		}
	}


	for (i=0;i<Ni;i++) {//Number of symbols=no. of col
		for (j=0;j<Ne;j++) {
			w[i][j] = new_weights[i+j*Ni];
		}
	}


	for (i=0;i<Ni;i++) {
		level2_I[i]=0; // 'current' is zero for o/p neurons at start
	}
	//n=0;

	/*initialization of v, R, T, H, v0 for every symbol testing*/

	// maximal conductance (in units of mS/cm^2)
	float C = 0.2;
	float v_K = -84;
	float g_K = 8;
	float v_Ca = 120;
	float g_Ca = 4.4;
	float v_leak = -60;
	float g_leak = 2;
	float v_1 = -1.2;
	float v_2 = 18;
	float v_3 = 2;
	float v_4 = 30;
	float phi = 0.04;

	//other morris variables for updating neurons at each time step

	float m_infty, w_infty, tau_w;

	// Time step for integration
	float dt=0.01; // for .01 recognition time 4.77, for .02, it is 4.68


	/*initialization of v, R, T, H, v0 for every symbol testing*/

	for (i=0;i<Ne;i++) {
		level1_v[i]=-60;
		level1_w_variable[i]=0.01;
		level1_v0[i]=0;
	}
	for (i=0;i<Ni;i++) {
		level2_v[i]=-60;
		level2_w_variable[i]=0.01;
		level2_v0[i]=0;
	}

	for (i=0;i<Ni;i++)
		level2_I[i]=0; // 'current' is zero for o/p neurons at start

	/************Testing starts*****************/

	start_clock=clock();

	for (loop=0;loop<iter;loop++) {
		for (t=1;t<1.15;t=t+dt) { //main simulation starts ;was 4.77
			
			//******************Level-2 Neurons Calculations************
			if (L1_firings[0]>0) {
				for (i=0;i<Ni;i++) {
					for (j=1;j<L1_firings[0]+1;j++) {
						level2_I[i]=level2_I[i]+w[i][L1_firings[j]];//weight add.
					}
				}
			}

			//Updating all output neurons
			for (i=0;i<Ni;i++) {
				if (level2_v0[i]<=30 && level2_v[i]>=30)
					printf("v[k]=%f, k=%d,t=%f\n",level2_v[i], i,t);
				level2_v0[i]=level2_v[i];

				m_infty =  0.5*(1+tanh((level2_v[i]-v_1)/v_2));
				w_infty = 0.5*(1+tanh((level2_v[i]-v_3)/v_4));
				tau_w   = 1/( cosh((level2_v[i]-v_3)/(2*v_4)) );

				level2_v[i] = level2_v[i] + (dt/C)*(level2_I[i]-g_Ca*m_infty*(level2_v[i]-v_Ca)-g_K * level2_w_variable[i] * (level2_v[i]-v_K)-g_leak * (level2_v[i]-v_leak));
				level2_w_variable[i] = level2_w_variable[i] + (dt/tau_w)*((w_infty-level2_w_variable[i])*phi);
			}
			
			//******************Level-1 Neurons Calculations************
			for (i=0;i<row;i++) {
				for (j=0;j<col;j++) {
					level1_I[j+col*i]=input[i][j]; //symbol ='current'
				}
			}
			L1_N_firings=1;

			//Upgrading all the input neurons
			for (i=0;i<Ne;i++) {
				if ((level1_v0[i]<=30) && (level1_v[i]>=30)) {
					L1_firings[L1_N_firings]=i;
					L1_N_firings++;
				}
				level1_v0[i]=level1_v[i];
				m_infty =  0.5*(1+tanh((level1_v[i]-v_1)/v_2));
				w_infty = 0.5*(1+tanh((level1_v[i]-v_3)/v_4));
				tau_w   = 1/( cosh((level1_v[i]-v_3)/(2*v_4)) );
				level1_v[i] = level1_v[i] + (dt/C)*(level1_I[i]-g_Ca*m_infty*(level1_v[i]-v_Ca)-g_K * level1_w_variable[i] * (level1_v[i]-v_K)-g_leak * (level1_v[i]-v_leak));
				level1_w_variable[i] = level1_w_variable[i] + (dt/tau_w)*((w_infty-level1_w_variable[i])*phi);
			}

			L1_firings[0]=L1_N_firings-1;
			for (i=0;i<Ni;i++)
				level2_I[i]=0;//initialization of the level-2 neurons for next time step
		}//end of t , simulation
		for (i=0;i<Ne;i++) {
			level1_v[i]=-60;
			level1_w_variable[i]=0.01;
			level1_v0[i]=0;
		}
		for (i=0;i<Ni;i++) {
			level2_v[i]=-60;
			level2_w_variable[i]=0.01;
			level2_v0[i]=0;
		}
	}//end of all iteration

	end_clock=clock();

	printf("\n Time required= %f  ms \n\n",
	(end_clock-start_clock)*1000/(CLOCKS_PER_SEC*iter));

	return 0;
}//end of main
