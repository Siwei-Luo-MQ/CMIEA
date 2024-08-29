from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re
import json
import pickle
import ast

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def straight_road_network(example_sketch,sketch,summary,record,folder_path,road_type):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n\n"
                                "It is known that this car crash happened on a straight road.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there in the road? (your answer here should be an integer)\n"
                                "2. How many lanes in total in the road? (your answer here should be an integer)\n"
                                "3. What's the length of the road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "4. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n\n"
                                "Your Output must in this structure: {'No_ways': <your answer for the number of ways>, 'No_lanes': <your answer for the number of lanes>, 'Length': <your answer for the length of the road>, 'Width': <your answer for the width of a single lane>}\n\n"
                                "Note: Your answers do not need to be enclosed in quotation marks\n\n"
                                "I will show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "\nSummary:\nVehicle one (V1 - case vehicle), a 1999 Ford Taurus, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way roadway. Vehicle two (V2), a 2002 Chevrolet Tahoe, 4-door utility vehicle was stopped at an angle, blocking the eastbound traffic lane. V2 is a marked police vehicle that had its overhead lights operating and the driver was out of the vehicle assisting other officers clear a previous crash. It was dark, the sky was clear, and the bituminous road was dry. The driver of V1 failed to completely clear the frost from her windshield and had a limited vision of the road. The driver of V1 did not see V2 and the front of V1 struck the right side of V2 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 66-year-old female driver of V1 (case occupant) was not using the available three-point seat belt and the steering-wheel airbag deployed. She was transported via ground ambulance to a local hosptial and later transferred to a regional level-one trauma center.\n\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from the crash summary\nFrom the summary, we can know that the following sentences described the number of ways and lanes:\n\n'Vehicle one (V1 - case vehicle), a 1999 Ford Taurus, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way roadway.'\nFrom this sentence, we can know that the word - 'two-way' is used for describing the number of ways and the word - 'two-lane' is used for describing the number of lanes.\nTherefor, the answer for the first question (number of ways) is 2 and the answer for the second question (number of lanes) is 2.\n\nStep 2 - Approximate the length of the road and the width of the lane from the sketch\nFrom the sketch we can know the two ends of the straight road are located on both sides of the sketch, so we need to calculate the distance from the leftmost end to the rightmost end in the sketch.\nThe red rectangular block marked with V1 in the image is a car. From the sketch, we can estimate that the length of the straight road is approximately equal to 11 car body lengths.\nTherefor we can calculate the length of the road in this way:\n5 x 11 = 55 meters\n\nFrom the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\nTherefor we can calculate the width of the lane in this way:\n2 + 2/2 + 2/2 = 4 meters\n\nFinally,  format the output:\n\n{'No_ways': 2, 'No_lanes': 2, 'Length': 55, 'Width': 4}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your answer for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    # Save original response
    # first_output = response.choices[0].message.content
    # network = json.loads(first_output)
    # network['Road type'] = road_type
    # file_name = f"{record}_road_network.pkl"
    # file_path = os.path.join(folder_path, file_name)
    # with open(file_path, "wb") as file:
    #     pickle.dump(network, file)
    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def curve_road_network(example_sketch,sketch,summary,record,folder_path,road_type,curvature):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n"
                                "It is known that this car crash happened on a curve road.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there in the road? (your answer here should be an integer)\n"
                                "2. How many lanes are in total on the road? (your answer here should be an integer)\n"
                                "3. What's the length of the road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "4. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "5. What's the curvature of the road? (your answer should be 'small' or 'big')\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n\n"
                                "Your Output must in this structure: {'No_ways': <your answer for the number of ways>, 'No_lanes': <your answer for the number of lanes>, 'Length': <your answer for the length of the road>, 'Width': <your answer for the width of a single lane>, 'Curvature': '<your answer for the curvature>'}\n\n"
                                "Note: Except for the curvature, your answers do not need to be enclosed in quotation marks.\n\nI will first show you a picture to explain the size of curvature and then show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{curvature}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "From the picture you can see two curves, one with a small curvature, which we named C1, and the other with a large curvature, which we named C2. If you think the curvature of the road in the sketch is more similar to C1, then the answer to Question 5 is small, but if you think the curvature of the road in the sketch is more similar to C2, then the answer to Question 5 is big."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Crash summary: \nVehicle one (V1 - case vehicle), a 1998 Mercury Mountaineer, 4-door utility vehicle was traveling east in the eastbound lane of a two-lane, two-way rural roadway and was negotiating a slight curve to the left. Vehicle two (V2), a 2002 Ford F150, 4-door extended cab pickup truck was traveling west in the westbound lane of the same road but was negoatiating a slight curve to the right. It was daylight, the sky was cloudy, and the bituminous road was dry. The driver of V2, for an unknown reason, drifted to the left, crossed the centerline, and entered the eastbound lane of travel. The driver of V1 attempted to avoid V2 by steering right but the front of V2 struck the front of V1 in a narrow, offset-frontal configuration. After the impact, V1 veered off the right side of the road and came to rest facing south. V2 was redirected back into the westbound lane, but then traveled back across the eastbound lane, departed the left side of the road, and came to rest off the south side of the road facing southwest. Both vehicles were towed due to disabling vehicle damage. The 34-year-old female driver of V1 (case occupant) was using the available three-point seat belt and the steering-wheel airbag deployed. The 2-year-old right-rear male passenger was seated in a forward-facing child seat that was secured to the vehicle using the available three-point seat belt. Both occupants of V1 were transported via helicopter to a regional level-one trauma center.\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from crash summary\nFrom the summary, we can know that the following sentences described the number of ways and lanes:\n\n'Vehicle one (V1 - case vehicle), a 1998 Mercury Mountaineer, 4-door utility vehicle was traveling east in the eastbound lane of a two-lane, two-way rural roadway and was negotiating a slight curve to the left. Vehicle two (V2), a 2002 Ford F150, 4-door extended cab pickup truck was traveling west in the westbound lane of the same road but was negoatiating a slight curve to the right.'\n\nFrom this sentence, we can know that these cars drive on the same road and the word - 'two-way' is used for describing the number of ways and the word - 'two-lane' is used for describing the number of lanes.\nTherefor, the answer for the first question (number of ways) is 2 and the answer for the second question (number of lanes) is 2.\n\nStep 2 - Approximate the length of the road and the width of the lane from the sketch\nWe define the straight-line distance between the two ends of a curve as the road length. From the sketch we can know the two ends of the curve road are located on both sides of the sketch, so we need to calculate the distance from the leftmost end to the rightmost end in the sketch.\nThe red rectangular block marked with V1 in the image is a car. From the sketch we can estimate that the straight-line distance between the two ends of this curve is approximately equal to 10 car body lengths.\nTherefor we can calculate the length of the road in this way:\nlength_road = 5 x 10 = 50 meters\n\nFrom the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\nTherefor we can calculate the width of the lane in this way:\nwidth_road = 2 + 2/2 + 2/2 = 4 meters\n\nStep 3 - Approximate the curvature\nComparing the curvature of the curved road in the sketch with C1 and C2, the curvature of the road in the sketch is obviously smaller than that of C1, so curvature = small\n\nFinally, Format the output:\n{'No_ways': 2, 'No_lanes': 2, 'Length': 50, 'Width': 4, 'Curvature': 'small'}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your answer for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    # Save original response
    # first_output = response.choices[0].message.content
    # network = json.loads(first_output)
    # network['Road type'] = road_type
    # file_name = f"{record}_road_network.pkl"
    # file_path = os.path.join(folder_path, file_name)
    # with open(file_path, "wb") as file:
    #     pickle.dump(network, file)
    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def intersection_road_network(example_sketch,sketch,summary,record,folder_path,road_type):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n"
                                "It is known that this car crash happened on an Intersection road.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there at most for roads bordering the intersection? (your answer here should be an integer)\n"
                                "2. How many lanes are there at most for roads bordering the intersection? (your answer here should be an integer)\n"
                                "3. What is the longest distance from one end of the intersection to the other? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "4. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n\n"
                                "Your Output must in this structure: {'No_ways': <your answer for the number of ways>, 'No_lanes': <your answer for the number of lanes>, 'Length': <your answer for the length of the road>, 'Width': <your answer for the width of a single lane>}\n\n"
                                "I will show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Crash summary: \nVehicle one (V1 - case vehicle), a 1997 Pontiac Grand Am, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way residential roadway and was approaching a 4-leg intersection. Vehicle two (V2), a 2002 Toyota Sequoia, 4-door utility vehicle was traveling north in the northbound lane of the intersecting two-lane, two-way road and was approaching the same intersection as V1. It was daylight, the sky was cloudy, and the bituminous road was dry. V1 and V2 entered the intersection at the same time and the front of V2 struck the right side of V1 in a T-type configuration. Both vehicles were driveable and both were driven away from the scene. The 28-year-old female driver of V1 was using the available three-point seat belt but the steering wheel airbag did not deploy. The 27-year-old female right-front passenger of V1 was not using the available three-point seat belt and the dash-mounted airbag did not deploy. The 7-year-old female left-rear passenger (case occupant) was improperly using the three-point seat belt by placing the shoulder belt under her arm. The 4-year-old male right-rear passenger was using the available three-point seat belt. The case occupant was transported via ground ambulance to a local hospital and later transferred to a regional level-one trauma center. The other three occupants of V1 did not seek medical attention.\n\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from crash summary\n\nFrom the summary, we can know that the following sentences described the number of ways and lanes:\n\n'Vehicle one (V1 - case vehicle), a 1997 Pontiac Grand Am, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way residential roadway and was approaching a 4-leg intersection. Vehicle two (V2), a 2002 Toyota Sequoia, 4-door utility vehicle was traveling north in the northbound lane of the intersecting two-lane, two-way road and was approaching the same intersection as V1.'\n\nFrom the above sentences, we can know that:\nV1 is driving on a two-lane, two-way residential roadway. Therefore, for the road where V1 is located, number_of_ways_v1 = 2, number_of_lanes_v1 = 2\nV2 is driving on a two-lane, two-way road. Therefore, for the road where V2 is located, number_of_ways_v2 = 2, number_of_lanes_v2 = 2\nFurthermore, for question 1, we need to select the largest value between number_of_ways_v1 and number_of_ways_v2, so the value of questions 1 is 2.\nFor question 2, we need to select the largest value between number_of_lanes_v2 and number_of_lanes_v2, so the value of questions 2 is 2.\n\n\nStep 2 - Approximate the longest distance from one end of the intersection to the other\n\nFrom the sketch, we know that the center of the intersection is roughly located at the center of the sketch. There are two roads in the sketch, one horizontal and one longitudinal, so we need to calculate the length of the road in the horizontal direction and the length of the road in the longitudinal direction, and compare the two, and take the longest one as the answer to question 3.\n\nFirst calculate the length of the road in the horizontal direction:\nThe red square marked with V1 in the horizontal direction represents a car. It is known that the length of the car is about 5 meters. From the figure, it can be estimated that the length of the road in the horizontal direction is about 8 car body lengths, so\nlength_horizontal_road = 5 x 8 = 40 meters\n\nNext, calculate the length of the road in the longitudinal direction:\nThe blue square marked with V2 in the longitudinal direction represents a car. It is known that the length of the car is about 5 meters. From the figure, it can be estimated that the length of the road in the longitudinal direction is about 7 car body lengths, so\nlength_longitudinal_road = 5 x 7 = 35 meters\n\nFor question 3, we need to select the largest value between length_horizontal_road and length_longitudinal_road, so the value of questions 3 is 40.\n\nStep 3 - Approximate the width of the lane from the sketch\n\nFrom the analysis results of Step 1, we know that this is a two-lane, two-way road, but the lane dividing line is not drawn in the sketch, so we cannot estimate the lane width based on the car width and lane shape. Therefore, for question 4, we can only give the default width of the lane, which is 4 meters.\n\nFinally, format the output:\n\n{'No_ways': 2, 'No_lanes': 2, 'Length': 40, 'Width': 4}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your output for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    # Save original response
    # first_output = response.choices[0].message.content
    # network = json.loads(first_output)
    # network['Road type'] = road_type
    # file_name = f"{record}_road_network.pkl"
    # file_path = os.path.join(folder_path, file_name)
    # with open(file_path, "wb") as file:
    #     pickle.dump(network, file)

    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def t_intersection_road_network(example_sketch,sketch,summary,record,folder_path,road_type):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n"
                                "It is known that this car crash happened on an T-Intersection road. (When two roads meet and form a T shape, it's called a T-intersection.) We call the horizontal lane in a T-intersection the main road, and the longitudinal lane the branch road. Vehicles heading towards the main road from the branch road need to give way to vehicles on the main road and turn.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there in the main road? (your answer here should be an integer)\n"
                                "2. How many lanes are there in the main road? (your answer here should be an integer)\n"
                                "3. How many ways are there in the branch road? (your answer here should be an integer)\n"
                                "4. How many lanes are there in the branch road? (your answer here should be an integer)\n"
                                "5. What is the length of the main road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "6. What is the length of the branch road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "7. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n\n"
                                "Your Output must be in this structure: {'No_ways_main_road': <your answer for the number of ways on the main road>, 'No_lanes_main_road': <your answer for the number of lanes on the main road>, 'No_ways_branch_road': <your answer for the number of ways on the branch road>, 'No_lanes_branch_road': <your answer for the number of lanes on the branch road>, 'Length_main': <your answer for the length of the main road>, 'Length_branch': <your answer for the length of the branch road>, 'Width': <your answer for the width of a single lane>}\n\n"
                                "Note: Your answers do not need to be enclosed in quotation marks.\n\n"
                                "I will show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Crash summary: Vehicle one (V1 - case vehicle), a 2001 Chrysler Sebring, 4-door sedan was traveling east in the eastbound left-turn lane of a three-lane, two-way private road that connects a company parking lot with a state roadway (one eastbound right-turn lane, one eastbound left-turn lane, and one westbound lane) and was approaching a T-intersection. Vehicle two (V2), a 1999 Chevrolet Blazer, 4-door utility vehicle was traveling south in the southbound through-lane of a three-lane, two-way road (one southbound right-turn lane, one southbound through-lane, and one northbound lane) and was approaching the same intersection as V1. It was daylight, the sky was clear, and the bituminous roads were dry. The driver of V1 attempted to turn left at the intersection and the front of V2 struck the left-side of V1 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 44-year-old male driver of V1 was using the available three-point seat belt and the steering-wheel airbag deployed. He was transported via helicopter to a regional level-one trauma center.\n\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from crash sketch & summary\n\nFrom the sketch, we can see that vehicle V1 is driving from the branch road to the main road, while V2 is driving on the main road. Therefore, the description of the driving section of V1 in the crash summary should include the description of the branch road features, while the description of the driving section of V2 should include the description of the main road features.\n\nIn the crash summary, the following statement describes V1:\n‘Vehicle one (V1 - case vehicle), a 2001 Chrysler Sebring, 4-door sedan was traveling east in the eastbound left-turn lane of a three-lane, two-way private road that connects a company parking lot with a state roadway (one eastbound right-turn lane, one eastbound left-turn lane, and one westbound lane) and was approaching a T-intersection.’\nFrom the above statement, we can know that the branch road is a three-lane, two-way private road, so for questions 3 and 4, our answers are:\nNo_ways_branch_road = 2\nNo_lanes_branch_road = 3\n\nIn the crash summary, the following statement describes V2:\n‘Vehicle two (V2), a 1999 Chevrolet Blazer, 4-door utility vehicle was traveling south in the southbound through-lane of a three-lane, two-way road (one southbound right-turn lane, one southbound through-lane, and one northbound lane) and was approaching the same intersection as V1.’\nFrom the above statement, we know that the main road is a three-lane, two-way road, so for questions 1 and 2, our answers are:\nNo_ways_main_road = 2\nNo_lanes_main_road = 3\n\n\nStep 2 - Approximate the length of the road and the width of the lane from the sketch\n\nFrom the sketch, we know that the center of the T-Intersection is roughly located at the center of the sketch. The main road is the horizontal road in the figure, and the branch road is the longitudinal road in the figure.\n\nFirst, calculate the length of the main road:\nFrom the sketch, we know that V2 is driving on the main road, and the length of the main road is the length from the leftmost end to the rightmost end of the sketch. Knowing that the length of the car body is 5 meters, it can be seen from the figure that the length of the main road is about 9 car body lengths, so the length of the main road is:\nlength_main_road = 5 x 9 = 45 meters\n\nNext, calculate the length of the branch road:\nFrom the sketch, we know that V1 is driving on the branch road, and the length of the branch road is the length from the top of the sketch to the center of the sketch. Knowing that the length of the car body is 5 meters, it can be seen from the figure that the length of the branch road is about 2 car body lengths, so the length of the main road is:\nlength_branch_road = 5 x 2 = 10 meters\n\nNext, estimate the width of the road:\nFrom the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\nTherefor we can calculate the width of the lane in this way:\nlane_width = 2 + 2/2 + 2/2 = 4 meters\n\nFinally, format the output:\n\n{'No_ways_main_road': 2, 'No_lanes_main_road': 3, 'No_ways_branch_road': 2, 'No_lanes_branch_road': 3, 'Length_main': 45, 'Length_branch': 10, 'Width': 4}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your output for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    # Save original response
    # first_output = response.choices[0].message.content
    # network = json.loads(first_output)
    # network['Road type'] = road_type
    # file_name = f"{record}_road_network.pkl"
    # file_path = os.path.join(folder_path, file_name)
    # with open(file_path, "wb") as file:
    #     pickle.dump(network, file)
    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def merge_road_network(example_sketch,sketch,summary,record,folder_path,road_type):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n"
                                "It is known that this car crash happened on a merge road. There is a main road and a branch road in the road network.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there on the main road? (your answer here should be an integer)\n"
                                "2. How many lanes are there on the main road? (your answer here should be an integer)\n"
                                "3. How many lanes are there on the branch road? (your answer here should be an integer)\n"
                                "4. What's the length of the main road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "5. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "6. What's the merge_angle of the road? (your answer should be 'small' or 'big')\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n"
                                "If the angle between the main road and the branch road is less than 45 degrees, the result of merge_angle is small, otherwise it is big.\n\n"
                                "Your Output must be in this structure: {'No_ways_main_road': <your answer for the number of ways on the main road>, 'No_lanes_main_road': <your answer for the number of lanes on the main road>, 'No_lanes_branch_road': <your answer for the number of lanes on the branch road>, 'Length_main': <your answer for the length of the main road>, 'Width': <your answer for the width of a single lane>, 'Merge angle': '<your answer for the angle of the merge point>'}\n\n"
                                "Note: Except for the Merge angle, your other answers do not need to be enclosed in quotation marks.\n\n"
                                "I will show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Crash summary: Vehicle one (V1 - case vehicle), a 2000 Pontiac Sunfire, 2-door coupe was traveling east in the right eastbound lane of a six-lane divided, limited access freeway (three eastbound travel lanes, protected median, three westbound lanes). It was daylight and raining, and the bituminous roadway was wet. Vehicle two (V2), a 1999 Dodge Ram conversion van, was traveling east in the eastbound lane of an entrance ramp and was attempting to enter the same freeway that V1 was traveling on. As V1 approached an area of mergance (end of an entrance ramp), the driver lost control due to standing water on the roadway. V1 began to rotate in a counterclockwise manner while it departed the right side of the road and entered onto the acceleration lane of the on-ramp. The driver of V2 attempted to stop but could not avoid striking the left-side of V1 with the front of V2. V1 was towed due to disabling vehicle damage and V2 was driven from the scene. The 26-year-old female driver of V1 (case occupant) was using the available three-point seat belt but no airbags deployed. She was transported to a local hospital and later transferred to a regional level-one trauma center. The driver of V2 was not injured and did not seek treatment.\n\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from crash summary\n\n"
                                "From the sketch, we can see that vehicle V1 is driving on the main road, while V2 is driving on the branch road and going to the mergance. Therefore, the description of the driving section of V1 in the crash summary should include the description of the main road features, while the description of the driving section of V2 should include the description of the branch road features.\n\n"
                                "In the crash summary, the following statement describes V1:\n"
                                "‘Vehicle one (V1 - case vehicle), a 2000 Pontiac Sunfire, 2-door coupe was traveling east in the right eastbound lane of a six-lane divided, limited access freeway (three eastbound travel lanes, protected median, three westbound lanes).’\n"
                                "From the above statement, we know that the main road is a six-lane, divided road, so for questions 1 and 2, our answers are:\n"
                                "No_ways_main_road = 2\n"
                                "No_lanes_main_road = 6\n\n"
                                "In the crash summary, the following statement describes V2:\n"
                                "‘Vehicle two (V2), a 1999 Dodge Ram conversion van, was traveling east in the eastbound lane of an entrance ramp and was attempting to enter the same freeway that V1 was traveling on.’\n"
                                "Unfortunately, there is no description of the number of branch roads in the above sentence, so we need to look at the sketch.\n"
                                "From the sketch, we can see that there is an obvious dotted line (lane dividing line) on the branch road where V2 is located. Therefore, the answer is 2 for questions 3.\n\n"
                                "Step 2 - Approximate the length of the main road and the width of the lane from the sketch\n"
                                "From the sketch we can know the two ends of the main road are located on both sides of the sketch, so we need to calculate the distance from the leftmost end to the rightmost end in the sketch.\n"
                                "The red rectangular block marked with V1 in the image is a car. From the sketch we can estimate that the straight-line distance between the two ends of this main road is approximately equal to 12 car body lengths.\n"
                                "Therefor we can calculate the length of the road in this way:\n"
                                "length_main_road = 5 x 12 = 60 meters\n\n"
                                "From the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\n"
                                "Therefor we can calculate the width of the lane in this way:\n"
                                "width_road = 2 + 2/2 + 2/2 = 4 meters\n\n"
                                "Step 3 - Approximate the merge_angle\n"
                                "From the sketch, we can see that the angle between the main road and the branch road is smaller than 45 degrees, so the answer for merge_angle is small.\n\n"
                                "Finally,  format the output:\n\n"
                                "{'No_ways_main_road': 2, 'No_lanes_main_road': 6, 'No_lanes_branch_road': 2, 'Length_main': 50, 'Width': 4, 'Merge angle': 'small'}\n"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your output for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    # Save original response
    # first_output = response.choices[0].message.content
    # network = json.loads(first_output)
    # network['Road type'] = road_type
    # file_name = f"{record}_road_network.pkl"
    # file_path = os.path.join(folder_path, file_name)
    # with open(file_path, "wb") as file:
    #     pickle.dump(network, file)
    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default=r'C:\Users\Kris\Desktop\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default=r'C:\Users\Kris\Desktop\CMIEA\road_type.xlsx', type=str,
                        help='Path of road type label file')
    args = parser.parse_args()

    # Get data ID list
    records = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]
    # Get road type labels
    df = pd.read_excel(args.road_type_label)
    road_type_labels = dict(zip(df['ID'], df['Label']))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"road_network_extract_experiment_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        record_id = int(record)
        record_path = args.data_path + '\\' + record
        # Get encoded crash sketch
        sketch = encode_image(record_path + '\\' + 'Sketch.jpg')
        # Get crash summary
        with open(record_path + '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()

        # Information Extraction Stage - Step 2
        # Extract road network from Summary and sketch
        if road_type_labels[record_id] == 'Straight':
            example_sketch = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\Straight\Sketch.jpg')
            straight_road_network(example_sketch,
                                  sketch,
                                  summary,
                                  record,
                                  folder_path,
                                  road_type_labels[record_id])
        elif road_type_labels[record_id] == 'Curve':
            example_sketch = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\Curve\Sketch.jpg')
            curvature = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\Curve\curvature.jpg')
            curve_road_network(example_sketch,
                                  sketch,
                                  summary,
                                  record,
                                  folder_path,
                                  road_type_labels[record_id],
                                  curvature
                               )
        elif road_type_labels[record_id] == 'Intersection':
            example_sketch = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\Intersection\Sketch.jpg')
            intersection_road_network(example_sketch,
                                  sketch,
                                  summary,
                                  record,
                                  folder_path,
                                  road_type_labels[record_id])
        elif road_type_labels[record_id] == 'T-intersection':
            example_sketch = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\T-Intersection\Sketch.jpg')
            t_intersection_road_network(example_sketch,
                                      sketch,
                                      summary,
                                      record,
                                      folder_path,
                                      road_type_labels[record_id])
        elif road_type_labels[record_id] == 'Merge':
            example_sketch = encode_image(
                r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\Road Attribute Extraction\Merge\Sketch.jpg')
            merge_road_network(example_sketch,
                                        sketch,
                                        summary,
                                        record,
                                        folder_path,
                                        road_type_labels[record_id])
        print(f'Record: {record} is finished!')


    data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            data_id = filename.split('_')[0]
            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                formatted_data = content[start:end]
                data = ast.literal_eval(formatted_data)
                data_dict[data_id] = data
            except (ValueError, SyntaxError):
                print(f"Can not find structure data in: {filename}")

    pkl_filename = folder_path + "\\road_network_results.pkl"
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(data_dict, pkl_file)

if __name__=='__main__':
    main()