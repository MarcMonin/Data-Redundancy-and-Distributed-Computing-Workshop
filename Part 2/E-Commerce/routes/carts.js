const express = require("express");
const Order = require("../models/Order");
const Product = require("../models/Product");
const { v4: uuidv4 } = require("uuid");

const router = express.Router();

