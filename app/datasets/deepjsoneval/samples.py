"""
DeepJSONEval 벤치마크용 샘플 데이터.

DeepJSONEval 논문(arxiv:2509.25922)의 패턴을 참고하여
deep-nested JSON 추출 성능을 평가하기 위한 4개 샘플.
각 샘플은 다양한 도메인, 다양한 nesting depth(3~5)를 가진다.
"""

from __future__ import annotations

SAMPLES: dict[str, dict] = {
    # ── Sample 1: Patient (depth 4) ──
    "sample_01_patient": {
        "category": "patient",
        "true_depth": 4,
        "text": (
            "Maria Garcia is a 45-year-old female patient who was admitted to "
            "St. Mary's General Hospital on January 15, 2025. Her blood type is AB-negative. "
            "She can be reached at +44-7700-900123 or via email at maria.garcia@example.org. "
            "Her emergency contact is her husband, Carlos Garcia, reachable at +44-7700-900456.\n\n"
            "Maria has a documented allergy to Penicillin, which causes severe anaphylaxis, "
            "and a moderate allergy to Shellfish that results in skin rash. "
            "She is currently taking two medications: Metformin at a dosage of 500mg twice daily "
            "(currently active) for diabetes management, and Ibuprofen at 200mg as needed "
            "(not currently active, was prescribed for post-surgical pain in 2023).\n\n"
            "Her medical history includes a diagnosis of Type 2 Diabetes on 2018-03-20, "
            "managed by Dr. Sarah Chen at St. Mary's Endocrinology Department. "
            "She also had an Appendectomy performed on 2023-07-10 by Dr. James Wilson "
            "at the General Surgery Department, which has been fully resolved.\n\n"
            "Recent vital signs recorded on 2025-01-15 show blood pressure at 128/82 mmHg, "
            "heart rate of 76 bpm, body temperature of 36.8 degrees Celsius, "
            "and oxygen saturation of 98 percent."
        ),
        "ground_truth": {
            "PersonalInfo": {
                "BasicInfo": {
                    "Name": "Maria Garcia",
                    "Gender": "Female",
                    "BloodType": "AB-"
                },
                "ContactInfo": {
                    "PhoneNumber": "+44-7700-900123",
                    "Email": "maria.garcia@example.org"
                },
                "EmergencyContact": {
                    "Name": "Carlos Garcia",
                    "Relationship": "Husband",
                    "PhoneNumber": "+44-7700-900456"
                }
            },
            "MedicalHistory": {
                "Allergies": [
                    {"AllergyName": "Penicillin", "Severity": "Severe", "Reaction": "Anaphylaxis"},
                    {"AllergyName": "Shellfish", "Severity": "Moderate", "Reaction": "Skin rash"}
                ],
                "Medications": [
                    {"MedicationName": "Metformin", "Dosage": 500, "Unit": "mg", "IsActive": True},
                    {"MedicationName": "Ibuprofen", "Dosage": 200, "Unit": "mg", "IsActive": False}
                ],
                "Diagnoses": [
                    {
                        "Condition": "Type 2 Diabetes",
                        "DiagnosedDate": "2018-03-20",
                        "Doctor": {"Name": "Dr. Sarah Chen", "Department": "Endocrinology"},
                        "Status": "Ongoing"
                    },
                    {
                        "Condition": "Appendectomy",
                        "DiagnosedDate": "2023-07-10",
                        "Doctor": {"Name": "Dr. James Wilson", "Department": "General Surgery"},
                        "Status": "Resolved"
                    }
                ]
            },
            "VitalSigns": {
                "RecordDate": "2025-01-15",
                "BloodPressure": "128/82",
                "HeartRate": 76,
                "Temperature": 36.8,
                "OxygenSaturation": 98
            }
        },
    },

    # ── Sample 2: Movie (depth 3) ──
    "sample_02_movie": {
        "category": "movie",
        "true_depth": 3,
        "text": (
            "宫崎骏导演的巅峰之作《千与千寻》（Spirited Away）堪称动画电影史上的璀璨明珠。"
            "这部2001年上映的奇幻动画由吉卜力工作室制作，片长125分钟，讲述了少女千寻在神灵世界"
            "的冒险与成长。\n\n"
            "在各大评分平台上，该片均获得了极高的评价。IMDb电影资料库的百万用户评分体系中，"
            "该片斩获8.6分。烂番茄（Rotten Tomatoes）网站上，该片获得97分的新鲜度评分。"
            "Metacritic综合评分高达96分。\n\n"
            "该片的配音阵容豪华：柊瑠美（Rumi Hiiragi）为千寻配音，入野自由（Miyu Irino）"
            "为白龙（Haku）配音，夏木真理（Mari Natsuki）则为汤婆婆（Yubaba）配音。\n\n"
            "该片荣获2003年第75届奥斯卡最佳动画长片奖（Academy Award for Best Animated Feature），"
            "以及2002年柏林国际电影节金熊奖（Golden Bear at Berlin International Film Festival），"
            "成为首部也是唯一获得金熊奖的动画电影。"
        ),
        "ground_truth": {
            "Title": "Spirited Away",
            "Director": "Hayao Miyazaki",
            "Year": 2001,
            "Runtime": 125,
            "Studio": "Studio Ghibli",
            "Ratings": [
                {"Source": "IMDb", "Score": 8.6},
                {"Source": "Rotten Tomatoes", "Score": 97},
                {"Source": "Metacritic", "Score": 96}
            ],
            "Cast": [
                {"Actor": "Rumi Hiiragi", "Character": "Chihiro"},
                {"Actor": "Miyu Irino", "Character": "Haku"},
                {"Actor": "Mari Natsuki", "Character": "Yubaba"}
            ],
            "Awards": [
                {
                    "Name": "Academy Award for Best Animated Feature",
                    "Year": 2003,
                    "Ceremony": "75th Academy Awards"
                },
                {
                    "Name": "Golden Bear",
                    "Year": 2002,
                    "Ceremony": "Berlin International Film Festival"
                }
            ]
        },
    },

    # ── Sample 3: Student Record (depth 5) ──
    "sample_03_student": {
        "category": "student",
        "true_depth": 5,
        "text": (
            "Student Profile: Zhang Wei (张伟), Student ID: 2021-CS-0042, is a senior undergraduate "
            "enrolled in the Computer Science program at Tsinghua University, School of Information "
            "Science and Technology. Date of birth: 1999-06-15. Email: zhangwei@tsinghua.edu.cn.\n\n"
            "Academic Record:\n"
            "Fall 2023 semester (GPA: 3.85/4.0):\n"
            "- CS401 Machine Learning, taught by Prof. Li Ming, Grade: A (4.0), Credits: 4. "
            "Course project: 'Transformer-based Image Classification' scored 92/100 with "
            "comment 'Excellent implementation and thorough analysis'.\n"
            "- CS450 Distributed Systems, taught by Prof. Wang Fang, Grade: A- (3.7), Credits: 3. "
            "Course project: 'Raft Consensus Implementation' scored 88/100 with "
            "comment 'Good understanding of consensus protocols'.\n\n"
            "Spring 2024 semester (GPA: 3.90/4.0):\n"
            "- CS460 Computer Vision, taught by Prof. Chen Jie, Grade: A (4.0), Credits: 4. "
            "Course project: 'Real-time Object Detection on Edge Devices' scored 95/100 with "
            "comment 'Outstanding work with practical applications'.\n"
            "- CS480 Natural Language Processing, taught by Prof. Liu Wei, Grade: A (4.0), Credits: 3. "
            "Course project: 'Chinese Sentiment Analysis using BERT' scored 90/100 with "
            "comment 'Well-designed experiments'.\n\n"
            "Research Experience:\n"
            "- Research Assistant at Tsinghua AI Lab from 2023-09 to 2024-06, supervised by "
            "Prof. Li Ming. Project: 'Efficient Fine-tuning of Large Language Models'. "
            "Publication: 'LoRA-Plus: Adaptive Low-Rank Adaptation' accepted at AAAI 2025 "
            "as second author.\n\n"
            "Awards: First Prize in National College Programming Contest 2023, "
            "Dean's List for Academic Excellence in Fall 2023 and Spring 2024."
        ),
        "ground_truth": {
            "StudentInfo": {
                "Name": "Zhang Wei",
                "StudentID": "2021-CS-0042",
                "DateOfBirth": "1999-06-15",
                "Email": "zhangwei@tsinghua.edu.cn",
                "University": "Tsinghua University",
                "School": "School of Information Science and Technology",
                "Program": "Computer Science",
                "Year": "Senior"
            },
            "AcademicRecord": {
                "Semesters": [
                    {
                        "Name": "Fall 2023",
                        "GPA": 3.85,
                        "Courses": [
                            {
                                "CourseCode": "CS401",
                                "CourseName": "Machine Learning",
                                "Instructor": "Prof. Li Ming",
                                "Grade": "A",
                                "GradePoint": 4.0,
                                "Credits": 4,
                                "Project": {
                                    "Title": "Transformer-based Image Classification",
                                    "Score": 92,
                                    "Comment": "Excellent implementation and thorough analysis"
                                }
                            },
                            {
                                "CourseCode": "CS450",
                                "CourseName": "Distributed Systems",
                                "Instructor": "Prof. Wang Fang",
                                "Grade": "A-",
                                "GradePoint": 3.7,
                                "Credits": 3,
                                "Project": {
                                    "Title": "Raft Consensus Implementation",
                                    "Score": 88,
                                    "Comment": "Good understanding of consensus protocols"
                                }
                            }
                        ]
                    },
                    {
                        "Name": "Spring 2024",
                        "GPA": 3.90,
                        "Courses": [
                            {
                                "CourseCode": "CS460",
                                "CourseName": "Computer Vision",
                                "Instructor": "Prof. Chen Jie",
                                "Grade": "A",
                                "GradePoint": 4.0,
                                "Credits": 4,
                                "Project": {
                                    "Title": "Real-time Object Detection on Edge Devices",
                                    "Score": 95,
                                    "Comment": "Outstanding work with practical applications"
                                }
                            },
                            {
                                "CourseCode": "CS480",
                                "CourseName": "Natural Language Processing",
                                "Instructor": "Prof. Liu Wei",
                                "Grade": "A",
                                "GradePoint": 4.0,
                                "Credits": 3,
                                "Project": {
                                    "Title": "Chinese Sentiment Analysis using BERT",
                                    "Score": 90,
                                    "Comment": "Well-designed experiments"
                                }
                            }
                        ]
                    }
                ]
            },
            "Research": [
                {
                    "Role": "Research Assistant",
                    "Lab": "Tsinghua AI Lab",
                    "StartDate": "2023-09",
                    "EndDate": "2024-06",
                    "Supervisor": "Prof. Li Ming",
                    "ProjectTitle": "Efficient Fine-tuning of Large Language Models",
                    "Publication": {
                        "Title": "LoRA-Plus: Adaptive Low-Rank Adaptation",
                        "Venue": "AAAI 2025",
                        "AuthorPosition": "Second author"
                    }
                }
            ],
            "Awards": [
                {"Name": "First Prize in National College Programming Contest", "Year": 2023},
                {"Name": "Dean's List for Academic Excellence", "Year": 2023},
                {"Name": "Dean's List for Academic Excellence", "Year": 2024}
            ]
        },
    },

    # ── Sample 4: Game (depth 4) ──
    "sample_04_game": {
        "category": "game",
        "true_depth": 4,
        "text": (
            "Stellar Odyssey is an epic space exploration RPG developed by Nova Interactive "
            "and published by Cosmos Entertainment. It was released on November 8, 2023 "
            "for PC, PlayStation 5, and Xbox Series X. The game is rated ESRB T for Teen.\n\n"
            "Players have spent an average of 63.7 hours exploring the vast universe. "
            "The game features several challenging achievements: 'Lunar Pioneer' requires "
            "players to establish the first outpost on the dark side of the moon; "
            "'Quantum Entangler' challenges players to solve all paradoxes in the quantum realm "
            "sequence; and 'Nebula Cartographer' demands mapping 100 percent of the Andromeda "
            "Sector anomalies.\n\n"
            "The game has received three major DLC expansions: 'Void Expansion Pack' released on "
            "2024-03-15 priced at $19.99, featuring 20 new star systems; "
            "'Chrono Splicers' released on 2024-07-22 priced at $14.99, adding time-travel "
            "mechanics with 15 new missions; and 'Echoes of Eridanus' released on 2024-11-30 "
            "priced at $24.99, the largest expansion with a new story arc and 30 hours of "
            "additional content.\n\n"
            "Critical reception has been very positive: IGN awarded it 9.2 out of 10, "
            "GameSpot gave it 9.0 out of 10, and PC Gamer rated it 88 out of 100. "
            "The user score on Metacritic stands at 8.7."
        ),
        "ground_truth": {
            "GameInfo": {
                "Title": "Stellar Odyssey",
                "Developer": "Nova Interactive",
                "Publisher": "Cosmos Entertainment",
                "ReleaseDate": "2023-11-08",
                "Platforms": ["PC", "PlayStation 5", "Xbox Series X"],
                "Rating": "ESRB T"
            },
            "GameStatistics": {
                "AveragePlaytime": 63.7,
                "Achievements": [
                    {
                        "Name": "Lunar Pioneer",
                        "Description": "Establish first outpost on the dark side of the moon"
                    },
                    {
                        "Name": "Quantum Entangler",
                        "Description": "Solve all paradoxes in the quantum realm sequence"
                    },
                    {
                        "Name": "Nebula Cartographer",
                        "Description": "Map 100% of the Andromeda Sector anomalies"
                    }
                ]
            },
            "DLCs": [
                {
                    "Name": "Void Expansion Pack",
                    "ReleaseDate": "2024-03-15",
                    "Price": 19.99,
                    "Description": "20 new star systems"
                },
                {
                    "Name": "Chrono Splicers",
                    "ReleaseDate": "2024-07-22",
                    "Price": 14.99,
                    "Description": "Time-travel mechanics with 15 new missions"
                },
                {
                    "Name": "Echoes of Eridanus",
                    "ReleaseDate": "2024-11-30",
                    "Price": 24.99,
                    "Description": "New story arc and 30 hours of additional content"
                }
            ],
            "Reviews": [
                {"Source": "IGN", "Score": 9.2, "MaxScore": 10},
                {"Source": "GameSpot", "Score": 9.0, "MaxScore": 10},
                {"Source": "PC Gamer", "Score": 88, "MaxScore": 100},
                {"Source": "Metacritic User", "Score": 8.7, "MaxScore": 10}
            ]
        },
    },
}
