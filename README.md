# Hotel_booking_prediction
This repository contains code and data for predicting hotel booking outcomes based on various features related to hotel reservations. The goal of this project is to build a predictive model that can help hotels anticipate whether a booking is likely to be canceled or not.
Dataset
The dataset used in this project contains information about hotel bookings. It includes details such as booking lead time, customer demographics, reservation status, and more. The dataset is provided in CSV format and can be found in the data/ directory.
Features
The following features are used in this project for predicting hotel booking outcomes:

Hotel Type (H1 or H2): Indicates whether the hotel is a Resort Hotel (H1) or a City Hotel (H2).
Is Canceled: A binary value indicating if the booking was canceled (1) or not (0).
Lead Time: The number of days that elapsed between the booking entry date and the arrival date.
Arrival Date Year: The year of arrival date.
Arrival Date Month: The month of arrival date.
Arrival Date Week Number: The week number of the year for the arrival date.
Arrival Date Day of Month: The day of the arrival date.
Stays in Weekend Nights: The number of weekend nights (Saturday or Sunday) the guest stayed.
Stays in Week Nights: The number of week nights (Monday to Friday) the guest stayed.
Adults: Number of adults.
Children: Number of children.
Babies: Number of babies.
Meal: Type of meal booked (e.g., BB, HB, FB).
Country: The country of origin of the guest.
Market Segment: Market segment designation (e.g., TA, TO).
Distribution Channel: Booking distribution channel (e.g., TA, TO).
Is Repeated Guest: A binary value indicating if the booking is from a repeated guest (1) or not (0).
Previous Cancellations: Number of previous bookings canceled by the customer before the current booking.
Previous Bookings Not Canceled: Number of previous bookings not canceled by the customer before the current booking.
Reserved Room Type: Code of room type reserved.
Assigned Room Type: Code for the type of room assigned to the booking.
Booking Changes: Number of changes/amendments made to the booking.
Deposit Type: Indication of whether a deposit was made (No Deposit, Non Refund, Refundable).
Agent: ID of the travel agency that made the booking.
Company: ID of the company/entity that made the booking.
Days in Waiting List: Number of days the booking was in the waiting list before confirmation.
Customer Type: Type of booking (e.g., Contract, Group, Transient).
ADR (Average Daily Rate): The average daily rate.
Required Car Parking Spaces: Number of car parking spaces required by the customer.
Total of Special Requests: Number of special requests made by the customer.
Reservation Status: Reservation last status (e.g., Canceled, Check-Out, No-Show).
Reservation Status Date: Date at which the last status was set.
