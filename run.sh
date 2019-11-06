#!/bin/bash
# Bash Menu Script Example
echo "######## ThinkerFarm Trainer ########"
PS3='Select an option: '
options=("Install" "Clean Traning Data" "Generate train and test labels csv" "Generate TFRecords" "Start Training" "Convert Model to Tflite" "Quit")
echo "~~~~~~~~~~~~~~~~~~~~~"
echo " M A I N - M E N U"
echo "~~~~~~~~~~~~~~~~~~~~~"
echo "1. Install Dependecies"
echo "2. Clean Traning Data"
echo "3. Generate train and test labels csv"
echo "4. Generate TFRecords"
echo "5. Start Training"
echo "6. Convert Model to Tflite"
echo "7. Quit"
echo "~~~~~~~~~~~~~~~~~~~~~"
select opt in "${options[@]}"
do
    case $opt in
        "Install")
            echo "~~~~~~~~~~~~~~~~~~~~~~~"
            echo "Installing dependencies"
            echo "~~~~~~~~~~~~~~~~~~~~~~~"
            ;;
        "Clean Traning Data")
            echo "~~~~~~~~~~~~~~~~~~~~~"
            echo "Cleaning Traning Data"
            echo "~~~~~~~~~~~~~~~~~~~~~"
            while true; do
              read -p "All your check points and training records will be deleted are you sure?" yn
              case $yn in
                [Yy]* )
                rm trainer/annotations/train.record
                rm trainer/annotations/test.record
                rm trainer/annotations/test_labels.csv
                rm trainer/annotations/test_images/*
                rm trainer/annotations/train_images/*
                rm trainer/annotations/train_labels.csv
                rm trainer/trained_check_points/*
                rm trainer/converted_model_tflite/*
                echo "~~~~~~~~~~~~~~~~~~~~~"
                echo "All traning data has been deleted"
                echo "~~~~~~~~~~~~~~~~~~~~~" ; break;;
                [Nn]* ) break;;
                *) echo "Please answer yes or no.";;
              esac
            done
            ;;

        "Generate train and test labels csv")


            train_labels=trainer/annotations/train_labels.csv

            if [ -f $train_labels ]; then
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm found $train_labels --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            else
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm Genereting $train_labels Please Wait... --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              python3 core/tfapi/xml_to_csv.py -i trainer/annotations/train_images -o trainer/annotations/train_labels.csv
            fi

            test_labels=trainer/annotations/test_labels.csv

            if [ -f $test_labels ]; then
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm found $test_labels --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            else
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm Genereting $test_labels Please Wait... --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              python3 core/tfapi/xml_to_csv.py -i trainer/annotations/test_images -o trainer/annotations/test_labels.csv
            fi
            ;;
        "Generate TFRecords")
            train_record=trainer/annotations/train.record

            if [ -f $train_record ]; then
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm found $train_record --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            else
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm Genereting $train_record Please Wait... --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              python3 core/tfapi/generate_tfrecord.py --label_map=trainer/annotations/label_map.pbtxt --csv_input=trainer/annotations/train_labels.csv --output_path=trainer/annotations/train.record --img_path=trainer/annotations/train_images
            fi

            test_record=trainer/annotations/test.record

            if [ -f $test_record ]; then
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm found $train_record --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            else
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              echo "-- ThinkerFarm Genereting $train_record Please Wait... --"
              echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              python3 core/tfapi/generate_tfrecord.py --label_map=trainer/annotations/label_map.pbtxt --csv_input=trainer/annotations/test_labels.csv --output_path=trainer/annotations/test.record --img_path=trainer/annotations/test_images
            fi
            ;;
        "Start Training")
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            echo "-- Starting training session --"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            python3 core/tfapi/train.py --logtostderr --train_dir=trainer/trained_check_points --pipeline_config_path=trainer/pre_trained_mobilenet/ssd_mobilenet_v2_quantized_300x300_coco.config

            ;;
        "Convert Model to Tflite")

            IFS=$'\r\n' GLOBIGNORE='*' command eval  'XYZ=($(cat trainer/trained_check_points/checkpoint))'
            echo "${XYZ[0]}"
            temp_dic=${XYZ[0]}
            temp_expr=${temp_dic::-1}
            temp_expr=${temp_expr:24}

            echo $temp_expr

            rm trainer/converted_model_tflite/*

            python3  core/tfapi/export_tflite_ssd_graph.py \
            --pipeline_config_path=trainer/pre_trained_mobilenet/ssd_mobilenet_v2_quantized_300x300_coco.config \
            --trained_checkpoint_prefix=trainer/trained_check_points/$temp_expr  \
            --output_directory=trainer/converted_model_tflite \
            --add_postprocessing_op=true

            tflite_convert \
                --output_file=trainer/converted_model_tflite/retrained_graph.tflite \
                --graph_def_file=trainer/converted_model_tflite/tflite_graph.pb \
                --input_arrays=normalized_input_image_tensor \
                --output_arrays="TFLite_Detection_PostProcess","TFLite_Detection_PostProcess:1","TFLite_Detection_PostProcess:2","TFLite_Detection_PostProcess:3" \
                --inference_type=FLOAT \
                --inference_input_type=QUANTIZED_UINT8 \
                --input_shapes=1,300,300,3 \
                --mean_values=128 \
                --std_dev_values=128 \
                --default_ranges_min=0 \
                --default_ranges_max=255 \
                --change_concat_input_ranges=false \
                --allow_custom_ops

            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            echo " TFLite model created "
            echo " please use retrained_graph.tflite file "
            echo " in trainer/converted_model_tflit folder"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
