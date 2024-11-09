#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs);
void update_parameters_adagrad(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs);
void update_parameters_adam(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);
void validate_gradients(unsigned int sample);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;

double final_learning_rate = 0.0001; // Final learning rate after decay
double alpha_m = 0.1; // Momentum decay parameter

// Validating gradients
double mismatch_avg_w_LI_L1 = 0;
double mismatch_avg_w_L1_L2 = 0;
double mismatch_avg_w_L2_L3 = 0;
double mismatch_avg_w_L3_LO = 0;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;

    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            // validate_gradients(training_sample);
            mean_loss+=obj_func;

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
            }
        }
        
        // Update weights on batch completion
        update_parameters(batch_size, epoch_counter, total_iter);
    }
    
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);

    // Print final validation of gradients
    // printf("Mismatch average w_LI_L1: %f\n", mismatch_avg_w_LI_L1);
    // printf("Mismatch average w_L1_L2: %f\n", mismatch_avg_w_L1_L2);
    // printf("Mismatch average w_L2_L3: %f\n", mismatch_avg_w_L2_L3);
    // printf("Mismatch average w_L3_LO: %f\n", mismatch_avg_w_L3_LO);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);

    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);

    // Evaluate parameter updates
    store_gradient_contributions();

    return loss;
}

void update_parameters(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs){
    // Part I To-do

    // Calculate learning rate decay
    double alpha = (double) epoch_counter / (double) total_epochs; // decay factor
    double current_learning_rate = learning_rate * (1 - alpha) + final_learning_rate * alpha; // decayed learning rate

    // Calculate and store η * (1/m) since it is constant for all weights
    // double learning_rate_div_m = learning_rate / batch_size; // For SGD without learning rate decay
    double learning_rate_div_m = current_learning_rate / batch_size; // For SGD with learning rate decay

    // Mini-batch Stochastic Gradient Descent
    // Batch SGD equation: w = w - η * (1/m) * ∑(i=1 to m) ∇Li(xi, w)
    // Define the updating for the weight matrices: w_LI_L1, w_L1_L2, w_L2_L3, w_L3_LO
    /*
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            w_LI_L1[i][j].w -= learning_rate_div_m * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            w_L1_L2[i][j].w -= learning_rate_div_m * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            w_L2_L3[i][j].w -= learning_rate_div_m * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j = 0; j < N_NEURONS_LO; j++){
            w_L3_LO[i][j].w -= learning_rate_div_m * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].dw = 0.0;
        }
    }
    */

    // Mini-batch Stochastic Gradient Descent with Momentum and Learning Rate Decay
    /*
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            w_LI_L1[i][j].v = alpha_m * w_LI_L1[i][j].v - learning_rate_div_m * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].w += w_LI_L1[i][j].v;
            w_LI_L1[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            w_L1_L2[i][j].v = alpha_m * w_L1_L2[i][j].v - learning_rate_div_m * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].w += w_L1_L2[i][j].v;
            w_L1_L2[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            w_L2_L3[i][j].v = alpha_m * w_L2_L3[i][j].v - learning_rate_div_m * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].w += w_L2_L3[i][j].v;
            w_L2_L3[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j = 0; j < N_NEURONS_LO; j++){
            w_L3_LO[i][j].v = alpha_m * w_L3_LO[i][j].v - learning_rate_div_m * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].w += w_L3_LO[i][j].v;
            w_L3_LO[i][j].dw = 0.0;
        }
    }
    */

    // update_parameters_adagrad(batch_size, epoch_counter, total_epochs);
    update_parameters_adam(batch_size, epoch_counter, total_epochs);
}

void update_parameters_adagrad(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs){
    // Calculate learning rate decay
    double alpha = (double) epoch_counter / (double) total_epochs; // decay factor
    double current_learning_rate = learning_rate * (1 - alpha) + final_learning_rate * alpha; // decayed learning rate

    // Calculate and store η * (1/m) since it is constant for all weights
    // double learning_rate_div_m = learning_rate / batch_size; // For SGD without learning rate decay
    double learning_rate_div_m = current_learning_rate / batch_size; // For SGD with learning rate decay

    // Implementing AdaGrad (TENDS TO GET STUCK IN THE LOCAL MINIMA AS WELL BUT DOES WORK AFTER A FEW TRIES)
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            w_LI_L1[i][j].p += w_LI_L1[i][j].dw * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].w -= (current_learning_rate / (sqrt(w_LI_L1[i][j].p) + 1e-8)) * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            w_L1_L2[i][j].p += w_L1_L2[i][j].dw * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].w -= (current_learning_rate / (sqrt(w_L1_L2[i][j].p) + 1e-8)) * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j <
        N_NEURONS_L3; j++){
            w_L2_L3[i][j].p += w_L2_L3[i][j].dw * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].w -= (current_learning_rate / (sqrt(w_L2_L3[i][j].p) + 1e-8)) * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j = 0; j < N_NEURONS_LO; j++){
            w_L3_LO[i][j].p += w_L3_LO[i][j].dw * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].w -= (current_learning_rate / (sqrt(w_L3_LO[i][j].p) + 1e-8)) * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].dw = 0.0;
        }
    }
}

void update_parameters_adam(unsigned int batch_size, unsigned int epoch_counter, unsigned int total_epochs){
    // Calculate learning rate decay
    double alpha = (double) epoch_counter / (double) total_epochs; // decay factor
    double current_learning_rate = learning_rate * (1 - alpha) + final_learning_rate * alpha; // decayed learning rate

    // Constants used for Adam
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    // Implementing Adam (DOES NOT WORK AND TENDS TO GET STUCK IN LOCAL MINIMA)
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            // Update moment estimates
            w_LI_L1[i][j].adam_m = beta1 * w_LI_L1[i][j].adam_m + (1 - beta1) * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].adam_v = beta2 * w_LI_L1[i][j].adam_v + (1 - beta2) * w_LI_L1[i][j].dw * w_LI_L1[i][j].dw;

            double beta = (current_learning_rate * sqrt(1 - pow(beta2, total_epochs))) / (1 - pow(beta1, total_epochs));

            // Update weights
            w_LI_L1[i][j].w -= (beta * w_LI_L1[i][j].adam_m) / (sqrt(w_LI_L1[i][j].adam_v) + epsilon);
            w_LI_L1[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            // Update moment estimates
            w_L1_L2[i][j].adam_m = beta1 * w_L1_L2[i][j].adam_m + (1 - beta1) * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].adam_v = beta2 * w_L1_L2[i][j].adam_v + (1 - beta2) * w_L1_L2[i][j].dw * w_L1_L2[i][j].dw;

            double beta = (current_learning_rate * sqrt(1 - pow(beta2, total_epochs))) / (1 - pow(beta1, total_epochs));

            // Update weights
            w_L1_L2[i][j].w -= (beta * w_L1_L2[i][j].adam_m) / (sqrt(w_L1_L2[i][j].adam_v) + epsilon);
            w_L1_L2[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            // Update moment estimates
            w_L2_L3[i][j].adam_m = beta1 * w_L2_L3[i][j].adam_m + (1 - beta1) * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].adam_v = beta2 * w_L2_L3[i][j].adam_v + (1 - beta2) * w_L2_L3[i][j].dw * w_L2_L3[i][j].dw;

            double beta = (current_learning_rate * sqrt(1 - pow(beta2, total_epochs))) / (1 - pow(beta1, total_epochs));

            // Update weights
            w_L2_L3[i][j].w -= (beta * w_L2_L3[i][j].adam_m) / (sqrt(w_L2_L3[i][j].adam_v) + epsilon);
            w_L2_L3[i][j].dw = 0.0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j = 0; j < N_NEURONS_LO; j++){
            // Update moment estimates
            w_L3_LO[i][j].adam_m = beta1 * w_L3_LO[i][j].adam_m + (1 - beta1) * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].adam_v = beta2 * w_L3_LO[i][j].adam_v + (1 - beta2) * w_L3_LO[i][j].dw * w_L3_LO[i][j].dw;

            double beta = (current_learning_rate * sqrt(1 - pow(beta2, total_epochs))) / (1 - pow(beta1, total_epochs));

            // Update weights
            w_L3_LO[i][j].w -= (beta * w_L3_LO[i][j].adam_m) / (sqrt(w_L3_LO[i][j].adam_v) + epsilon);
            w_L3_LO[i][j].dw = 0.0;
        }
    }
}

void validate_gradients(unsigned int val_training_sample){
    double mismatch_count = 0;
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            double w = w_LI_L1[i][j].w;
            double dw = w_LI_L1[i][j].dw;
            double epsilon = 1e-6;
            w_LI_L1[i][j].w = w + epsilon;
            double loss_plus = compute_xent_loss(training_labels[val_training_sample]);
            w_LI_L1[i][j].w = w - epsilon;
            double loss_minus = compute_xent_loss(training_labels[val_training_sample]);
            w_LI_L1[i][j].w = w;
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
            if (fabs(dw - numerical_gradient) > 0.0001){
                mismatch_count++;
            }
        }
    }
    if (mismatch_avg_w_LI_L1 == 0){
        mismatch_avg_w_LI_L1 = mismatch_count / (N_NEURONS_LI * N_NEURONS_L1);
    } else {
        mismatch_avg_w_LI_L1 = (mismatch_avg_w_LI_L1 + (mismatch_count / (N_NEURONS_LI * N_NEURONS_L1))) / 2;
    }
    mismatch_count = 0;

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            double w = w_L1_L2[i][j].w;
            double dw = w_L1_L2[i][j].dw;
            double epsilon = 1e-6;
            w_L1_L2[i][j].w = w + epsilon;
            double loss_plus = compute_xent_loss(training_labels[val_training_sample]);
            w_L1_L2[i][j].w = w - epsilon;
            double loss_minus = compute_xent_loss(training_labels[val_training_sample]);
            w_L1_L2[i][j].w = w;
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
            if (fabs(dw - numerical_gradient) > 0.0001){
                mismatch_count++;
            }
        }
    }    
    if (mismatch_avg_w_L1_L2 == 0){
        mismatch_avg_w_L1_L2 = mismatch_count / (N_NEURONS_L1 * N_NEURONS_L2);
    } else {
        mismatch_avg_w_L1_L2 = (mismatch_avg_w_L1_L2 + (mismatch_count / (N_NEURONS_L1 * N_NEURONS_L2))) / 2;
    }
    mismatch_count = 0;

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            double w = w_L2_L3[i][j].w;
            double dw = w_L2_L3[i][j].dw;
            double epsilon = 1e-6;
            w_L2_L3[i][j].w = w + epsilon;
            double loss_plus = compute_xent_loss(training_labels[val_training_sample]);
            w_L2_L3[i][j].w = w - epsilon;
            double loss_minus = compute_xent_loss(training_labels[val_training_sample]);
            w_L2_L3[i][j].w = w;
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
            if (fabs(dw - numerical_gradient) > 0.0001){
                mismatch_count++;
            }
        }
    }
    if (mismatch_avg_w_L2_L3 == 0){
        mismatch_avg_w_L2_L3 = mismatch_count / (N_NEURONS_L2 * N_NEURONS_L3);
    }
    else{
        mismatch_avg_w_L2_L3 = (mismatch_avg_w_L2_L3 + (mismatch_count / (N_NEURONS_L2 * N_NEURONS_L3))) / 2;
    }
    mismatch_count = 0;

    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j = 0; j < N_NEURONS_LO; j++){
            double w = w_L3_LO[i][j].w;
            double dw = w_L3_LO[i][j].dw;
            double epsilon = 1e-6;
            w_L3_LO[i][j].w = w + epsilon;
            double loss_plus = compute_xent_loss(training_labels[val_training_sample]);
            w_L3_LO[i][j].w = w - epsilon;
            double loss_minus = compute_xent_loss(training_labels[val_training_sample]);
            w_L3_LO[i][j].w = w;
            double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
            if (fabs(dw - numerical_gradient) > 0.0001){
                mismatch_count++;
            }
        }
    }
    if (mismatch_avg_w_L3_LO == 0){
        mismatch_avg_w_L3_LO = mismatch_count / (N_NEURONS_L3 * N_NEURONS_LO);
    }
    else{
        mismatch_avg_w_L3_LO = (mismatch_avg_w_L3_LO + (mismatch_count / (N_NEURONS_L3 * N_NEURONS_LO))) / 2;
    }
    mismatch_count = 0;
}
